import re
import gc
import random
import arc_agi
import numpy as np
from scipy.ndimage import label
from preprocess import TrajectoryDataset
from config import COT_TOKENS, END_MARK, logger
from typing import Any, Dict, List, Optional, Tuple
from generator import InteractiveGenerator, format_trajectory

class ARC3Env:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.official_env = arc_agi.Arcade().make(task_id)
        self.local_game = None

    def _hook_local_game(self) -> Any:
        if self.local_game is not None:
            return self.local_game
        
        for obj in gc.get_objects():
            if type(obj).__name__.lower() == self.task_id.lower():
                self.local_game = obj
                return obj
        return None

    def reset(self) -> str:
        obs = self.official_env.reset()
        self._hook_local_game()
        
        for _ in range(5):
            grid = self._get_best_grid()
            if grid is not None and len(grid) > 0 and len(grid) < 60 and len(grid[0]) < 60:
                break
            
            try:
                action_to_send = 1
                if hasattr(self.official_env, "action_space"):
                    for act in self.official_env.action_space:
                        if getattr(act, "value", None) == 1 or "1" in str(act):
                            action_to_send = act
                            break
                self.official_env.step(action_to_send)
            except Exception:
                pass
        
        return self._format_observation()

    def step(self, action_obj: Any) -> Tuple[str, float, bool, Dict]:
        try:
            step_result = self.official_env.step(action_obj)
        except Exception as e:
            logger.error(f"Official env rejected action {action_obj}: {e}")
            return self._format_observation(), 0.0, False, {}
        
        if isinstance(step_result, tuple):
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
        else:
            reward = 0.0
            done = False
            info = {}
        
        return self._format_observation(), float(reward), done, info

    def _format_observation(self) -> str:
        grid = self._get_best_grid()
        
        if not grid or len(grid) == 0:
            return "Grid:\nEmpty Grid\n\n=== OBJECT MANIFEST ===\nNo objects detected."
        
        np_grid = np.array(grid)
        counts = np.bincount(np_grid.flatten())
        bg_color = np.argmax(counts)
        
        lines = []
        for row in grid:
            line = [f" {str(int(val))} " if val != bg_color and val != 0 else " . " for val in row]
            lines.append("".join(line))
        grid_text = "\n".join(lines)
        
        manifest_text = self._generate_object_manifest(grid, bg_color)
        return f"Grid:\n{grid_text}\n\n{manifest_text}"

    def _get_best_grid(self) -> List[List[int]]:
        target = self._hook_local_game()
        if not target:
            return []
        
        logical_grid = self._extract_logical_grid(target)
        if logical_grid is not None:
            return logical_grid.tolist()
        
        logger.debug("Logical grid not found; falling back to auto-cropped camera view.")
        cropped_grid = self._crop_camera_noise(target)
        if cropped_grid is not None:
            return cropped_grid.tolist()
        
        return []

    def _extract_logical_grid(self, target) -> np.ndarray | None:
        candidates = [target]
        if hasattr(target, "game"):
            candidates.append(target.game)
        
        for obj in candidates:
            for attr in ("grid", "board", "matrix"):
                val = getattr(obj, attr, None)
                if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                    arr = np.array(val)
                    if arr.ndim == 2:
                        return arr
            
            cur = getattr(obj, "current_level", None)
            if cur is not None:
                for attr in ("grid", "board", "matrix", "cells", "puzzle"):
                    val = getattr(cur, attr, None)
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        arr = np.array(val)
                        if arr.ndim == 2:
                            return arr
            
            levels = getattr(obj, "levels", None)
            if isinstance(levels, (list, tuple)) and len(levels) > 0:
                idx = getattr(obj, "level_index", 0)
                level = levels[idx] if idx < len(levels) else levels[0]
                for attr in ("grid", "board", "matrix", "cells", "puzzle"):
                    val = getattr(level, attr, None)
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        arr = np.array(val)
                        if arr.ndim == 2:
                            return arr
        return None

    def _crop_camera_noise(self, target) -> np.ndarray | None:
        if hasattr(target, "game"):
            target = target.game
        
        cam = getattr(target, "camera", None)
        if cam is None:
            return None
        
        w = getattr(cam, "width", 64)
        h = getattr(cam, "height", 64)
        
        try:
            pixels = target.get_pixels(0, 0, w, h)
        except Exception:
            return None
        
        if not isinstance(pixels, np.ndarray):
            pixels = np.array(pixels)
        
        unique, counts = np.unique(pixels, return_counts=True)
        if len(unique) == 0:
            return pixels
        
        bg = unique[np.argmax(counts)]
        mask = pixels != bg
        
        if not np.any(mask):
            return pixels
        
        coords = np.argwhere(mask)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        
        r_min = max(0, r_min - 1)
        r_max = min(h - 1, r_max + 1)
        c_min = max(0, c_min - 1)
        c_max = min(w - 1, c_max + 1)
        
        return pixels[r_min : r_max + 1, c_min : c_max + 1]

    def _generate_object_manifest(self, grid: List[List[int]], bg_color: int) -> str:
        np_grid = np.array(grid)
        unique_colors = np.unique(np_grid)
        
        manifest_lines = ["=== OBJECT MANIFEST ==="]
        structure = np.ones((3, 3), dtype=int)
        global_obj_idx = 1
        
        for color in unique_colors:
            if color == bg_color or color == 0:
                continue
            
            mask = (np_grid == color)
            labeled_array, num_features = label(mask, structure)
            
            for local_idx in range(1, num_features + 1):
                coords = np.argwhere(labeled_array == local_idx)
                mass = len(coords)
                min_r, min_c = coords.min(axis=0)
                max_r, max_c = coords.max(axis=0)
                
                manifest_lines.append(
                    f"Object_{global_obj_idx}: Color {color}, Mass {mass}, "
                    f"Bounds (R:{min_r}-{max_r}, C:{min_c}-{max_c})"
                )
                global_obj_idx += 1
        
        if global_obj_idx == 1:
            return "=== OBJECT MANIFEST ===\nNo distinct objects detected."
        
        return "\n".join(manifest_lines)

DEFAULT_FEW_SHOT_ARC3 = """You are an AI agent playing a puzzle game from the ARC-AGI-3 benchmark.
Your task is to observe the grid environment and its Object Manifest, infer the underlying logical rules, and take actions to complete the puzzle.

Example interaction:
Game: ls20
Observation:
Grid:
 .  .  1  .  .
 .  1  1  1  .
 .  .  1  .  .
 .  .  .  .  .
 .  .  2  2  .
 .  .  2  2  .

=== OBJECT MANIFEST ===
Object_1: Color 1, Mass 5, Bounds (R:0-2, C:1-3)
Object_2: Color 2, Mass 4, Bounds (R:4-5, C:2-3)

Available actions: 1, 2, 3, 4, 5

Think: I see two distinct objects. Action 1 might manipulate Object_1. Let's test it.
Act: 1

Observation:
Grid:
 .  .  .  .  .
 .  .  1  .  .
 .  .  .  .  .
 .  .  .  .  .
 .  .  2  2  .
 .  .  2  2  .

=== OBJECT MANIFEST ===
Object_1: Color 1, Mass 1, Bounds (R:1-1, C:2-2)
Object_2: Color 2, Mass 4, Bounds (R:4-5, C:2-3)

Available actions: 1, 2, 3, 4, 5

Think: Action 1 reduced Object_1's mass from 5 to 1. I need to continue manipulating the objects to reach the goal.
Act: 2
...
Goal achieved! The puzzle is complete.

Important: If you try an action and the state does not change, do NOT repeat it endlessly. Analyze the manifest and move on to another action.
"""

class ARC3Generator(InteractiveGenerator):
    def __init__(self, config: Dict[str, Any], dataset_config: Dict[str, Any], args=None):
        super().__init__(config, dataset_config, args)
        self._fallback_available = False

    def setup_environment(self, task: Dict[str, Any], split: str = "train") -> ARC3Env:
        game_id = task.get("game_id")
        if not game_id:
            raise ValueError("Task dictionary must contain 'game_id'")

        try:
            env = ARC3Env(task_id=game_id)
            logger.info(f"Successfully loaded hybrid ARC3Env for: {game_id}")
            return env
        except Exception as e:
            logger.error(f"Failed to load hybrid ARC3Env {game_id}: {e}")
            raise

    def load_builtin_tasks(self) -> List[Dict[str, Any]]:
        config = self.dataset_config
        game_ids = config.get("game_ids", [])

        tasks = []
        for gid in game_ids:
            tasks.append({
                "id": f"arc3_{gid}",
                "game_id": gid,
                "instruction": f"You are playing the ARC-AGI-3 puzzle '{gid}'. Observe the grid and object manifest, infer the logic, and solve the task."
            })

        if not tasks:
            default_id = "ls20"
            logger.warning(f"No game IDs; defaulting to '{default_id}'.")
            tasks.append({
                "id": f"arc3_{default_id}",
                "game_id": default_id,
                "instruction": f"You are playing the ARC-AGI-3 puzzle '{default_id}'. Observe the grid and object manifest, infer the logic, and solve the task."
            })
        return tasks

    def build_react_prompt(
        self,
        observation: str,
        goal: str,
        admissible_commands: List[str],
        history: List[Dict[str, str]],
        few_shot_example: str = ""
    ) -> str:
        if not few_shot_example:
            few_shot_example = DEFAULT_FEW_SHOT_ARC3

        base_prompt = super().build_react_prompt(
            observation, goal, admissible_commands, history, few_shot_example
        )

        goal_reminder = (
            "!!! REMINDER: Your ONLY task is:\n"
            f"Goal: {goal}\n"
            "Ignore any distracting information. Focus exclusively on the Goal above.\n\n"
        )

        planning_reminder = (
            "Before acting, form a simple plan:\n"
            "1. Take one action and observe how the Object Manifest changes to learn the rules.\n"
            "2. Use that knowledge to manipulate the objects toward the win state.\n"
            "3. If an action does nothing, do NOT repeat it endlessly. Move on to another action.\n\n"
        )

        prompt = base_prompt.replace(
            "You are an AI assistant playing a text-based game.",
            goal_reminder + planning_reminder + "\nYou are an AI assistant playing a text-based game."
        )

        if self.use_cot_tokens:
            token_descs = [
                f"<{n}> ... </{n}> : {c['description']}" if END_MARK else f"<{n}>: ... : {c['description']}"
                for n, c in COT_TOKENS.items()
            ]
            token_instructions = (
                "\n\nIMPORTANT: Structure your reasoning using the following special tokens:\n"
                + "\n".join(token_descs)
                + "\nWrap each reasoning step with the appropriate token. "
            )
            token_instructions += (
                "Use exactly <token> content </token> format." if END_MARK else "Use exactly <token>: content format."
            )
            token_instructions += "\nThe final action must still be preceded by 'Act:'.\nExample Think block: Think: \n"
            
            token_names = list(COT_TOKENS.keys())
            if len(token_names) >= 2:
                if END_MARK:
                    example = f"<{token_names[0]}> some content </{token_names[0]}>\n<{token_names[1]}> more content </{token_names[1]}>"
                else:
                    example = f"<{token_names[0]}>: some content\n<{token_names[1]}>: more content"
            else:
                example = "reasoning content"
            
            token_instructions += example
            
            prompt = prompt.replace(
                "Think: [Your reasoning]",
                "Think: [Your reasoning, using the special tokens as instructed]"
            ) + token_instructions

        return prompt

    def parse_action_from_response(self, response: str, admissible_commands: List[str]) -> Optional[str]:
        flat = list(admissible_commands)
        lines = response.split("\n")

        for line in reversed(lines):
            m = re.search(r'^(?:\*\*|__)?Act(?:\*\*|__)?:\s*(.+?)\s*$', line.strip(), re.IGNORECASE)
            if m:
                action = re.sub(r'[.!?,;:]$', '', m.group(1).strip())
                if action in flat:
                    return action

        non_empty = [l.strip() for l in lines if l.strip()]
        for line in reversed(non_empty[-3:]):
            clean = line.replace('`', '').strip()
            if clean in flat:
                return clean

        if non_empty:
            last = non_empty[-1].lower()
            for cmd in sorted(flat, key=len, reverse=True):
                if re.search(r'\b' + re.escape(cmd.lower()) + r'\b', last):
                    return cmd

        return None

    def _extract_think_metadata(self, response: str) -> Dict[str, Optional[str]]:
        m = re.search(
            r'(?:\*\*|__)?Think(?:\*\*|__)?:\s*(.+?)(?=\n\s*(?:\*\*|__)?Act|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        think = m.group(1).strip() if m else None
        meta = {"think": think}
        if think:
            for token in COT_TOKENS:
                pat = rf'<{token}>(.*?)</{token}>' if END_MARK else rf'<{token}>:\s*(.*?)(?=\n<|$)'
                m2 = re.search(pat, think, re.IGNORECASE | re.DOTALL)
                if m2:
                    meta[token] = m2.group(1).strip()
        return meta

    def _get_admissible_commands(self, env: ARC3Env) -> List[str]:
        if hasattr(env.official_env, "action_space"):
            try:
                if hasattr(env.official_env.action_space, "n"):
                    return [str(i) for i in range(env.official_env.action_space.n)]
                
                commands = []
                for a in env.official_env.action_space:
                    if hasattr(a, "value") and isinstance(a.value, int):
                        commands.append(str(a.value))
                    else:
                        commands.append(str(a).replace("GameAction.", ""))
                return commands
            except Exception:
                pass
        return ["1", "2", "3", "4", "5", "6"]

    def _string_to_action(self, env: ARC3Env, action_str: str) -> Any:
        if hasattr(env.official_env, "action_space"):
            try:
                for act in env.official_env.action_space:
                    if hasattr(act, "value") and str(act.value) == action_str:
                        return act
                    if str(act).replace("GameAction.", "") == action_str:
                        return act
                    if str(act) == action_str:
                        return act
            except Exception:
                pass
        
        if action_str.isdigit():
            return int(action_str)
        return action_str

    def run_episode(self, task: Dict[str, Any], split: str = "train") -> Dict[str, Any]:
        env = self.setup_environment(task, split)
        
        obs_text = env.reset()

        step_count = 0
        steps = []
        history = []
        goal = task.get("instruction", f"Complete the puzzle {task.get('game_id')}")

        done = False
        success = False

        while not done and step_count < self.max_steps:
            admissible_commands = self._get_admissible_commands(env)
            
            observation_payload = f"{obs_text}\n\nAvailable actions: {', '.join(admissible_commands)}"

            if self.show_prompt:
                logger.info(f"Step {step_count+1} observation:\n{observation_payload}")

            prompt = self.build_react_prompt(observation_payload, goal, admissible_commands, history)
            
            if self.show_prompt:
                logger.info(f"Prompt:\n{prompt}")

            response = self.llm_client.get_response(prompt)
            
            if self.show_response:
                logger.info(f"Response:\n{response}")

            action_str = self.parse_action_from_response(response, admissible_commands)
            think_meta = self._extract_think_metadata(response)

            if action_str is None:
                action_str = random.choice(admissible_commands)
                logger.warning(f"Invalid response, selecting random action: {action_str}")

            if self.show_action:
                logger.info(f"Action: {action_str}")

            action_obj = self._string_to_action(env, action_str)
            logger.debug(f"Sending action: {action_obj} (type: {type(action_obj)})")

            step_data = {
                "observation": observation_payload,
                "action": action_str,
                "admissible_commands": admissible_commands.copy()
            }
            step_data.update(think_meta)
            steps.append(step_data)
            history.append({"observation": observation_payload, "action": action_str})

            next_obs_text, reward, is_done, info = env.step(action_obj)
            
            obs_text = next_obs_text
            done = is_done
            step_count += 1

            if done or reward > 0:
                logger.info(f"Puzzle logic resolved! (Reward: {reward})")
                success = reward > 0
                done = True

        result = {
            "task_id": task.get("id", task.get("game_id")),
            "goal": goal,
            "game_id": task.get("game_id"),
            "success": success,
            "total_steps": step_count,
            "metadata": {"max_steps_reached": step_count >= self.max_steps and not done}
        }

        if self.output_format == "messages":
            result["messages"] = self.convert_trajectory_to_messages(steps, goal)
        else:
            result["trajectory"] = steps

        if not success and self.show_trajectory_on_fail:
            logger.info(
                "Trajectory for failed task %s:\n%s",
                task.get("id"),
                format_trajectory(steps, goal)
            )

        return result

    def convert_trajectory_to_messages(
        self, steps: List[Dict[str, Any]], goal: str
    ) -> List[Dict[str, str]]:
        messages = []
        for i, step in enumerate(steps):
            user_content = step["observation"]
            if i == 0:
                user_content = f"Goal: {goal}\n\nObservation:\n{user_content}"
            messages.append({"role": "user", "content": user_content})

            think = step.get("think", "")
            if think:
                assistant_content = f"Think: {think}\nAct: {step['action']}"
            else:
                assistant_content = f"Act: {step['action']}"
            messages.append({"role": "assistant", "content": assistant_content})
        return messages

class ARC3(TrajectoryDataset):
    INSTRUCTION = (
        "You are an AI agent playing a puzzle game from the ARC-AGI-3 benchmark. "
        "Explore the grid, infer the logic, and take actions to reach the win state. "
        "Based on the current matrix, object manifest, and history, determine the best next action."
    )

    def __init__(self, name: str, split: str, config):
        super().__init__(name, split, config)
        if config is None:
            from config import load_datasets_config
            config = load_datasets_config().get(name, {})
        self.dataset_config = config if not hasattr(config, 'dataset') else config.dataset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = self.data[idx]
        if self.output_format == "messages":
            msgs = ep.get("messages") or self.to_chat_format(idx)
            return {
                "id": ep.get("task_id", f"task_{idx}"),
                "task_id": ep.get("task_id", f"task_{idx}"),
                "goal": ep.get("goal", ""),
                "game_id": ep.get("game_id", ""),
                "messages": msgs,
                "success": ep.get("success", False)
            }
        return {
            "id": ep.get("task_id", f"task_{idx}"),
            "task_id": ep.get("task_id", f"task_{idx}"),
            "goal": ep.get("goal", ""),
            "game_id": ep.get("game_id", ""),
            "trajectory": ep.get("trajectory", []),
            "success": ep.get("success", False)
        }

    def to_chat_format(self, idx: int) -> List[Dict[str, str]]:
        ep = self.data[idx]
        if ep.get("messages"):
            return ep["messages"]
        msgs = []
        goal = ep.get("goal", "")
        for i, step in enumerate(ep.get("trajectory", [])):
            user = step["observation"]
            if i == 0:
                user = f"Goal: {goal}\n\nObservation:\n{user}"
            msgs.append({"role": "user", "content": user})
            think = step.get("think", "")
            assistant = f"Think: {think}\nAct: {step['action']}" if think else f"Act: {step['action']}"
            msgs.append({"role": "assistant", "content": assistant})
        return msgs

    def _create_temp_generator(self) -> ARC3Generator:
        g = ARC3Generator.__new__(ARC3Generator)
        g.config = {}
        g.dataset_config = self.dataset_config
        g.max_steps = self.max_steps
        g.mode = self.split
        g.use_cot_tokens = False
        return g

    def create_interactive_env(self, task: Dict[str, Any], split: str = "train") -> Any:
        return self._create_temp_generator().setup_environment(task, split)

    def parse_action(self, response: str, admissible_commands: List[str]) -> Optional[str]:
        return self._create_temp_generator().parse_action_from_response(response, admissible_commands)

    def build_prompt(
        self,
        observation: str,
        goal: str,
        admissible_commands: List[str],
        history: List[Dict[str, str]]
    ) -> str:
        return self._create_temp_generator().build_react_prompt(
            observation, goal, admissible_commands, history
        )
