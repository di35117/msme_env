"""
LinguisticDecodingEnvironment — Domain-agnostic RL environment server.

This class contains ZERO domain-specific logic.
All domain logic is delegated to the active DomainAdapter.

Lifecycle:
  reset()  → load adapter, generate world, build observation
  step()   → classify outcome, reward, propagate network effects,
              advance time step, build observation, inject memory
  state()  → return current episode state
"""

import time
from typing import Dict, List, Optional, Any

from openenv.core.env_server.base_environment import BaseEnvironment
from openenv.core.env_server.types import State

from ..core.models import LinguisticDecodingAction, LinguisticDecodingObservation
from ..domains import get_adapter, DomainAdapter

try:
    from ..core.memory import (
        MemorySystem,
        build_working_memory,
        retrieve_semantic_context,
        retrieve_episodic_context,
        update_semantic_memory,
        store_episodic_memory,
    )
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


class LinguisticDecodingEnvironment(
    BaseEnvironment[LinguisticDecodingAction, LinguisticDecodingObservation]
):
    """
    OpenEnv-compliant environment for linguistic decoding under hidden state.

    The environment is a thin orchestrator:
      1. Load domain adapter (by name, from registry)
      2. Delegate world generation, outcome classification, reward,
         network propagation, and message generation to the adapter
      3. Manage episode lifecycle and memory injection
      4. Return standardised observations and rewards
    """

    def __init__(self):
        # Runtime state — reset on each episode
        self._adapter: Optional[DomainAdapter] = None
        self._hidden_profiles: Dict[int, Dict] = {}
        self._episode_history: List[Dict] = []
        self._episode_histories: List[List[Dict]] = []  # across all episodes
        self._episode: int = 0
        self._time_step: int = 0
        self._episode_reward: float = 0.0
        self._active_alerts: List[str] = []
        self._memory: Optional[Any] = None

    # ------------------------------------------------------------------
    # OpenEnv API: reset
    # ------------------------------------------------------------------

    def reset(self, payload: Dict = None) -> LinguisticDecodingObservation:
        payload = payload or {}
        domain_id = payload.get("domain", "msme_startup")

        # Load adapter (cache if same domain)
        if self._adapter is None or self._adapter.domain_id != domain_id:
            self._adapter = get_adapter(domain_id)

        # Archive completed episode history
        if self._episode_history:
            self._episode_histories.append(self._episode_history)

        self._episode += 1
        self._time_step = 1
        self._episode_reward = 0.0
        self._episode_history = []
        self._active_alerts = []

        # Compute adversarial weaknesses from past episodes
        weaknesses = None
        if self._episode > 40 and self._episode_histories:
            weaknesses = self._adapter.analyze_weaknesses(
                self._episode_histories[-20:]  # last 20 episodes
            )

        # Generate world
        self._hidden_profiles = self._adapter.generate_world(
            episode=self._episode,
            adversarial_weaknesses=weaknesses,
        )

        # Initialise memory
        if MEMORY_AVAILABLE:
            self._memory = MemorySystem(domain_id=domain_id)

        return self._build_observation(
            last_action_result=None,
            step_reward=0.0,
        )

    # ------------------------------------------------------------------
    # OpenEnv API: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: LinguisticDecodingAction,
    ) -> tuple[LinguisticDecodingObservation, float, bool]:

        adapter = self._adapter
        if adapter is None:
            raise RuntimeError("Call reset() before step().")

        # Validate action type
        if action.action_type not in adapter.valid_actions:
            outcome = "malformed_json_format"
            step_reward = -0.15
        else:
            # Classify outcome from hidden state
            outcome = adapter.classify_outcome(
                action_type=action.action_type,
                target_id=action.target_id,
                hidden_profiles=self._hidden_profiles,
                time_step=self._time_step,
                parameters=action.parameters,
            )

            # Compute step reward
            step_reward = adapter.compute_step_reward(
                action_type=action.action_type,
                target_id=action.target_id,
                outcome=outcome,
                hidden_profiles=self._hidden_profiles,
            )

        self._episode_reward += step_reward

        # Propagate network effects
        self._hidden_profiles, new_alerts = adapter.propagate_network_effects(
            hidden_profiles=self._hidden_profiles,
            action_type=action.action_type,
            target_id=action.target_id,
            outcome=outcome,
        )
        self._active_alerts = new_alerts

        # Generate message (logged, not sent in obs — RM reviews separately)
        observable = next(
            (s for s in adapter.build_speakers_observation(
                self._hidden_profiles, self._time_step, self._episode_history
            ) if s.get("id") == action.target_id),
            {}
        )
        message = adapter.generate_message(
            action_type=action.action_type,
            target_id=action.target_id,
            hidden_profiles=self._hidden_profiles,
            observable=observable,
            parameters=action.parameters,
        )

        # Build step record for history and memory
        step_record = {
            "time_step":   self._time_step,
            "action_type": action.action_type,
            "target_id":   action.target_id,
            "account_type": self._hidden_profiles.get(
                action.target_id, {}
            ).get("account_type", "unknown"),
            "parameters":  action.parameters,
            "reasoning":   action.reasoning,
            "outcome":     outcome,
            "step_reward": step_reward,
            "message":     message,
            "alerts":      new_alerts,
        }
        self._episode_history.append(step_record)

        # Update memory
        if MEMORY_AVAILABLE and self._memory:
            store_episodic_memory(self._memory, step_record)
            update_semantic_memory(self._memory, step_record)

        # Advance time step
        self._hidden_profiles = adapter.advance_time_step(
            hidden_profiles=self._hidden_profiles,
            time_step=self._time_step,
            episode_history=self._episode_history,
        )
        self._time_step += 1

        # Check episode termination
        done = self._time_step > adapter.time_steps

        # Compute episode reward on termination
        episode_reward = None
        if done:
            result = adapter.compute_episode_reward(
                hidden_profiles=self._hidden_profiles,
                episode_history=self._episode_history,
                episode_num=self._episode,
            )
            episode_reward = result["total"]
            # Log component breakdown
            step_record["episode_reward_components"] = result

        # Build last_action_result for observation
        last_action_result = {
            "action_type": action.action_type,
            "target_id":   action.target_id,
            "outcome":     outcome,
            "message":     message,
            "step_reward": step_reward,
            "network_effects": new_alerts,
        }

        obs = self._build_observation(
            last_action_result=last_action_result,
            step_reward=step_reward,
        )

        reward = episode_reward if done else step_reward
        return obs, reward, done

    # ------------------------------------------------------------------
    # OpenEnv API: state
    # ------------------------------------------------------------------

    def get_state(self) -> State:
        return State(
            episode_id=self._episode,
            step_count=len(self._episode_history),
        )

    # ------------------------------------------------------------------
    # Private: build observation
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        last_action_result: Optional[Dict],
        step_reward: float,
    ) -> LinguisticDecodingObservation:
        adapter = self._adapter

        speakers = adapter.build_speakers_observation(
            self._hidden_profiles,
            self._time_step,
            self._episode_history,
        )
        portfolio_summary = adapter.build_portfolio_summary(
            self._hidden_profiles,
            self._time_step,
        )
        domain_state = adapter.build_domain_state(
            self._hidden_profiles,
            self._time_step,
        )

        # Memory injection
        working_memory = ""
        semantic_context = ""
        episodic_context = ""
        if MEMORY_AVAILABLE and self._memory:
            working_memory = build_working_memory(
                self._memory, portfolio_summary, self._episode_history
            )
            semantic_context = retrieve_semantic_context(
                self._memory, last_action_result
            )
            episodic_context = retrieve_episodic_context(
                self._memory, last_action_result
            )

        done = self._time_step > adapter.time_steps

        return LinguisticDecodingObservation(
            episode=self._episode,
            step=len(self._episode_history),
            time_step=self._time_step,
            speakers=speakers,
            portfolio_summary=portfolio_summary,
            working_memory=working_memory,
            semantic_memory_context=semantic_context,
            episodic_memory_context=episodic_context,
            last_action_result=last_action_result,
            active_network_alerts=self._active_alerts,
            domain_state=domain_state,
            step_reward=step_reward,
            episode_reward_so_far=self._episode_reward,
            done=done,
            reward=None,
            metadata={
                "domain":    adapter.domain_id,
                "time_step": self._time_step,
                "episode":   self._episode,
            },
        )
