# SPDX-License-Identifier: Apache-2.0
"""Fork runtime integration: use this server's address, key and state root."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omlx.transcription.runtime import runtime_directory
from omlx.transcription.translation import translation_connection


class RuntimeTests(unittest.TestCase):
    def settings(self, root='unused'):
        return SimpleNamespace(base_path=root, server=SimpleNamespace(port=18082),
                               auth=SimpleNamespace(api_key='test-only-key'))

    def test_default_auxiliary_models_use_same_server_with_authentication(self):
        with patch.dict(os.environ, {}, clear=True), patch('omlx.settings.get_settings', return_value=self.settings()):
            self.assertEqual(translation_connection(), ('http://127.0.0.1:18082/v1',
                             {'Authorization': 'Bearer test-only-key'}))

    def test_external_server_does_not_receive_local_api_key(self):
        with patch.dict(os.environ, {'TRANSLATION_BASE_URL': 'http://external.example/v1/'}, clear=True), patch(
                'omlx.settings.get_settings', return_value=self.settings()):
            self.assertEqual(translation_connection(), ('http://external.example/v1', {}))

    def test_explicit_external_key_is_used(self):
        with patch.dict(os.environ, {'TRANSLATION_BASE_URL': 'http://external.example/v1',
                'TRANSLATION_API_KEY': 'external-test-key'}, clear=True), patch(
                'omlx.settings.get_settings', return_value=self.settings()):
            self.assertEqual(translation_connection()[1], {'Authorization': 'Bearer external-test-key'})

    def test_state_is_created_under_configured_base_path(self):
        root=Path(__file__).resolve().parents[2]/'.artifacts/transcription-tests'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder, patch('omlx.settings.get_settings', return_value=self.settings(folder)):
            self.assertEqual(runtime_directory('uploads'), Path(folder)/'transcription/uploads')
            self.assertTrue(runtime_directory('uploads').is_dir())
