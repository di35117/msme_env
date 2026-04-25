"""
DomainAdapter — Abstract Base Class for Linguistic Decoding domains.

Every domain module must implement this interface.
The LinguisticDecodingEnvironment (server/environment.py) calls through
this interface exclusively — it contains zero domain-specific logic.

To add a new domain:
  1. Create linguistic_decoding/domains/<your_domain>/
  2. Implement DomainAdapter in adapter.py
  3. Register in linguistic_decoding/domains/__init__.py

The core engine handles: OpenEnv API contracts, memory injection,
reward bookkeeping, and episode lifecycle.

The adapter handles: world generation, observable state building,
outcome classification, step reward computation, network propagation,
and message generation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class DomainAdapter(ABC):
    """
    Abstract interface for a linguistic decoding domain.

    Type parameters (informal):
        HiddenProfile   — full hidden state for one speaker (dict)
        ObservableState — what the agent sees for one speaker (dict)
        ActionType      — string action names valid in this domain
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def domain_id(self) -> str:
        """
        Unique string identifier for this domain, e.g. "msme_startup".
        Used for routing in the server and for memory namespacing.
        """

    @property
    @abstractmethod
    def total_speakers(self) -> int:
        """Total number of speakers/accounts per episode."""

    @property
    @abstractmethod
    def time_steps(self) -> int:
        """
        Number of time steps per episode.
        For msme_startup: 36 (months).
        For other domains: however long an episode runs.
        """

    @property
    @abstractmethod
    def valid_actions(self) -> List[str]:
        """
        All valid action_type strings for this domain.
        The environment rejects unknown action types with a format penalty.
        """

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_world(
        self,
        episode: int,
        adversarial_weaknesses: Optional[Dict[str, float]] = None,
    ) -> Dict[int, Dict]:
        """
        Generate hidden profiles for all speakers in a new episode.

        Args:
            episode: Episode number (used for curriculum phasing).
            adversarial_weaknesses: Dict of {weakness_key: score} from
                analyze_weaknesses(). None for the first N episodes.

        Returns:
            Dict mapping speaker_id (int) → hidden_profile (dict).
            Hidden profiles must never be returned to the agent.
        """

    # ------------------------------------------------------------------
    # Observable state
    # ------------------------------------------------------------------

    @abstractmethod
    def build_speakers_observation(
        self,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
        episode_history: List[Dict],
    ) -> List[Dict]:
        """
        Build the list of observable speaker dicts from hidden profiles.

        Each dict must include:
          - id (int)
          - speaker_type (str) — domain-defined, e.g. "msme" / "startup"
          - last_message (str) — the speaker's most recent communication
          - behavioral_signals (dict) — observable proxy signals
          - observable_metrics (dict) — e.g. dpd, mrr, filing_status

        Hidden fields (true health, runway, etc.) must be stripped.

        Args:
            hidden_profiles: Full hidden state for all speakers.
            time_step: Current time step.
            episode_history: List of past steps in this episode.

        Returns:
            List of observable speaker state dicts.
        """

    @abstractmethod
    def build_portfolio_summary(
        self,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
    ) -> Dict:
        """
        Build the portfolio-level summary visible to the agent.
        May include aggregate DPD distribution, account type counts, etc.
        Must not reveal hidden state directly.
        """

    @abstractmethod
    def build_domain_state(
        self,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
    ) -> Dict:
        """
        Build the domain_state dict injected into the observation.
        For msme_startup: {'msme_accounts': [...], 'startup_accounts': [...]}.
        For other domains: whatever structured context the agent prompt needs.
        """

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    @abstractmethod
    def classify_outcome(
        self,
        action_type: str,
        target_id: int,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
        parameters: Dict[str, Any],
    ) -> str:
        """
        Simulate the outcome of an action given hidden state.

        Returns an outcome string that maps to a reward in compute_step_reward.
        The outcome must be deterministic given the hidden profile —
        no random draws inside here (use the hidden profile's pre-drawn
        random fields for stochastic elements).

        Args:
            action_type: The action being taken.
            target_id: Speaker being acted on.
            hidden_profiles: Full hidden state (not visible to agent).
            time_step: Current time step.
            parameters: Action-specific parameters.

        Returns:
            Outcome string, e.g. "payment_received_after_moratorium".
        """

    @abstractmethod
    def compute_step_reward(
        self,
        action_type: str,
        target_id: int,
        outcome: str,
        hidden_profiles: Dict[int, Dict],
    ) -> float:
        """
        Compute immediate reward for this action outcome.

        Args:
            action_type: The action taken.
            target_id: Speaker acted on.
            outcome: Output of classify_outcome.
            hidden_profiles: Full hidden state for context/amplification.

        Returns:
            Float reward (positive or negative).
        """

    # ------------------------------------------------------------------
    # Network effects
    # ------------------------------------------------------------------

    @abstractmethod
    def propagate_network_effects(
        self,
        hidden_profiles: Dict[int, Dict],
        action_type: str,
        target_id: int,
        outcome: str,
    ) -> Tuple[Dict[int, Dict], List[str]]:
        """
        Propagate network effects from this action across connected speakers.

        Args:
            hidden_profiles: Current hidden state (will be mutated in place).
            action_type: The action taken.
            target_id: Speaker acted on.
            outcome: Outcome string from classify_outcome.

        Returns:
            Tuple of (updated_hidden_profiles, list_of_alert_strings).
            Alert strings are injected into active_network_alerts in the
            next observation.
        """

    # ------------------------------------------------------------------
    # Episode reward
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_episode_reward(
        self,
        hidden_profiles: Dict[int, Dict],
        episode_history: List[Dict],
        episode_num: int,
    ) -> Dict[str, float]:
        """
        Compute the final episode reward from hard, verifiable numbers.
        No LLM judge. Returns a dict with component scores and 'total'.

        Args:
            hidden_profiles: Final hidden state after all steps.
            episode_history: Complete list of step records.
            episode_num: Episode number (for curriculum phasing).

        Returns:
            Dict with at minimum: {'total': float}.
            Additional keys (npa_rate, recovery_rate, etc.) are logged.
        """

    # ------------------------------------------------------------------
    # Weakness analysis (for adversarial curriculum)
    # ------------------------------------------------------------------

    @abstractmethod
    def analyze_weaknesses(
        self,
        episode_histories: List[List[Dict]],
    ) -> Dict[str, float]:
        """
        Analyze completed episodes to find agent weaknesses.
        Called by the adversarial curriculum controller.

        Args:
            episode_histories: List of episode histories (each a list of steps).

        Returns:
            Dict of {weakness_key: score}. Higher score = more exploitable.
            These scores are passed back into generate_world() on next reset.
        """

    # ------------------------------------------------------------------
    # Message generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_message(
        self,
        action_type: str,
        target_id: int,
        hidden_profiles: Dict[int, Dict],
        observable: Dict,
        parameters: Dict[str, Any],
    ) -> str:
        """
        Generate the outbound RM/agent message for this action.

        During simulation: uses deterministic templates.
        During training: replaced by LLM inference from the fine-tuned model.

        Args:
            action_type: Action being communicated.
            target_id: Speaker being messaged.
            hidden_profiles: Full hidden state (for template context).
            observable: Observable state for this speaker.
            parameters: Action parameters (e.g. months=2 for moratorium).

        Returns:
            Generated message string.
        """

    # ------------------------------------------------------------------
    # Update hidden state across time steps
    # ------------------------------------------------------------------

    @abstractmethod
    def advance_time_step(
        self,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
        episode_history: List[Dict],
    ) -> Dict[int, Dict]:
        """
        Advance hidden profiles to the next time step.

        Handles: trajectory evolution (health declining/improving),
        crisis trigger activation, NPA classification at DPD thresholds,
        and any time-based state transitions.

        Args:
            hidden_profiles: Current hidden state.
            time_step: The time step just completed.
            episode_history: All steps taken so far.

        Returns:
            Updated hidden profiles.
        """
