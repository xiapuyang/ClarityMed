"""Re-export shim — ProfileContextFeature moved to core.features.profile_context_plugin."""

from claritymed.core.features.profile_context_plugin import (
    ProfileContextFeature as ProfileContextFeature,
    _format_profile_block as _format_profile_block,
)

__all__ = ["ProfileContextFeature", "_format_profile_block"]
