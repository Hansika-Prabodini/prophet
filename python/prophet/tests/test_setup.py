# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for setup.py model compilation functionality.
Tests cover both successful compilation scenarios and edge cases.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest

# Add parent directory to path to import setup module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import setup


class TestBuildCmdStanModel:
    """Test suite for build_cmdstan_model function."""

    @patch("setup.install_cmdstan_deps")
    @patch("setup.copy")
    @patch("setup.copytree")
    @patch("setup.prune_cmdstan")
    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    @patch("setup.MODEL_DIR", "stan")
    @patch("setup.CMDSTAN_VERSION", "2.33.1")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_successful_model_compilation(
        self,
        mock_remove,
        mock_exists,
        mock_repackage,
        mock_prune,
        mock_copytree,
        mock_copy,
        mock_install_deps,
    ):
        """Test successful Stan model compilation."""
        # Setup mocks
        mock_repackage.return_value = True
        mock_exists.side_effect = lambda path: True  # All files exist
        
        # Mock cmdstanpy
        mock_model = Mock()
        mock_model.exe_file = "/tmp/prophet.exe"
        
        with patch("setup.cmdstanpy.CmdStanModel", return_value=mock_model):
            with tempfile.TemporaryDirectory() as target_dir:
                # Should not raise any exceptions
                setup.build_cmdstan_model(target_dir)
                
                # Verify install_cmdstan_deps was called
                assert mock_install_deps.called
                
                # Verify model file was copied
                assert mock_copy.call_count >= 2  # Stan file + executable
                
                # Verify prune was called
                assert mock_prune.called

    @patch("setup.install_cmdstan_deps")
    @patch("setup.copy")
    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    @patch("setup.MODEL_DIR", "stan")
    @patch("os.path.exists")
    def test_missing_stan_file(
        self,
        mock_exists,
        mock_repackage,
        mock_copy,
        mock_install_deps,
    ):
        """Test compilation fails when Stan file is missing."""
        mock_repackage.return_value = True
        
        # Make the Stan file not exist
        def exists_side_effect(path):
            if "prophet.stan" in str(path):
                return False
            return True
        
        mock_exists.side_effect = exists_side_effect
        
        with tempfile.TemporaryDirectory() as target_dir:
            with pytest.raises(RuntimeError, match="Failed to copy Stan model file"):
                setup.build_cmdstan_model(target_dir)

    @patch("setup.install_cmdstan_deps")
    @patch("setup.copy")
    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    @patch("setup.MODEL_DIR", "stan")
    @patch("os.path.exists")
    def test_compilation_failure(
        self,
        mock_exists,
        mock_repackage,
        mock_copy,
        mock_install_deps,
    ):
        """Test proper error handling when Stan model compilation fails."""
        mock_repackage.return_value = True
        mock_exists.return_value = True
        
        # Mock cmdstanpy to raise an exception during compilation
        with patch("setup.cmdstanpy.CmdStanModel", side_effect=Exception("Compilation error")):
            with tempfile.TemporaryDirectory() as target_dir:
                with pytest.raises(RuntimeError, match="Failed to compile Stan model"):
                    setup.build_cmdstan_model(target_dir)

    @patch("setup.install_cmdstan_deps")
    @patch("setup.copy")
    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    @patch("setup.MODEL_DIR", "stan")
    @patch("os.path.exists")
    def test_missing_executable_after_compilation(
        self,
        mock_exists,
        mock_repackage,
        mock_copy,
        mock_install_deps,
    ):
        """Test error when compiled executable is not found after compilation."""
        mock_repackage.return_value = True
        
        # Make executable not exist
        def exists_side_effect(path):
            if ".exe" in str(path) or "prophet_model" in str(path):
                return False
            return True
        
        mock_exists.side_effect = exists_side_effect
        
        mock_model = Mock()
        mock_model.exe_file = "/tmp/prophet.exe"
        
        with patch("setup.cmdstanpy.CmdStanModel", return_value=mock_model):
            with tempfile.TemporaryDirectory() as target_dir:
                with pytest.raises(RuntimeError, match="Compiled model executable not found"):
                    setup.build_cmdstan_model(target_dir)

    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    def test_cmdstan_installation_failure(
        self,
        mock_repackage,
    ):
        """Test error handling when CmdStan installation fails."""
        mock_repackage.return_value = True
        
        with patch("setup.install_cmdstan_deps", side_effect=RuntimeError("Installation failed")):
            with tempfile.TemporaryDirectory() as target_dir:
                with pytest.raises(RuntimeError, match="Failed to install CmdStan dependencies"):
                    setup.build_cmdstan_model(target_dir)

    @patch("setup.install_cmdstan_deps")
    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    @patch("setup.MODEL_DIR", "stan")
    @patch("os.path.exists")
    def test_file_copy_failure(
        self,
        mock_exists,
        mock_repackage,
        mock_install_deps,
    ):
        """Test error handling when file copy operations fail."""
        mock_repackage.return_value = True
        mock_exists.return_value = True
        
        with patch("setup.copy", side_effect=IOError("Permission denied")):
            with tempfile.TemporaryDirectory() as target_dir:
                with pytest.raises(RuntimeError, match="Failed to copy"):
                    setup.build_cmdstan_model(target_dir)

    @patch("setup.install_cmdstan_deps")
    @patch("setup.copy")
    @patch("setup.copytree")
    @patch("setup.prune_cmdstan")
    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", True)
    @patch("setup.MODEL_DIR", "stan")
    @patch("setup.CMDSTAN_VERSION", "2.33.1")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_windows_specific_behavior(
        self,
        mock_remove,
        mock_exists,
        mock_repackage,
        mock_prune,
        mock_copytree,
        mock_copy,
        mock_install_deps,
    ):
        """Test Windows-specific compilation behavior."""
        mock_repackage.return_value = True
        mock_exists.return_value = True
        
        mock_model = Mock()
        mock_model.exe_file = "C:\\tmp\\prophet.exe"
        
        with patch("setup.cmdstanpy.CmdStanModel", return_value=mock_model):
            with tempfile.TemporaryDirectory() as target_dir:
                setup.build_cmdstan_model(target_dir)
                
                # Verify copytree was called for Windows to copy cmdstan directory
                assert mock_copytree.called

    @patch("setup.install_cmdstan_deps")
    @patch("setup.copy")
    @patch("setup.prune_cmdstan")
    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    @patch("setup.MODEL_DIR", "stan")
    @patch("os.path.exists")
    def test_cleanup_process(
        self,
        mock_exists,
        mock_repackage,
        mock_prune,
        mock_copy,
        mock_install_deps,
    ):
        """Test that cleanup process runs and handles errors gracefully."""
        mock_repackage.return_value = True
        mock_exists.return_value = True
        
        mock_model = Mock()
        mock_model.exe_file = "/tmp/prophet.exe"
        
        # Create a temporary directory structure to simulate cleanup
        with tempfile.TemporaryDirectory() as tmp_dir:
            stan_dir = Path(tmp_dir) / "stan"
            stan_dir.mkdir()
            
            # Create dummy files for cleanup
            (stan_dir / "prophet.stan").touch()
            (stan_dir / "temp_file.hpp").touch()
            (stan_dir / "temp_file.o").touch()
            
            with patch("setup.MODEL_DIR", str(stan_dir)):
                with patch("setup.cmdstanpy.CmdStanModel", return_value=mock_model):
                    with tempfile.TemporaryDirectory() as target_dir:
                        setup.build_cmdstan_model(target_dir)
                        
                        # Verify cleanup was attempted
                        # The .stan file should remain, others should be removed
                        assert (stan_dir / "prophet.stan").exists()

    @patch("setup.install_cmdstan_deps")
    @patch("setup.copy")
    @patch("setup.prune_cmdstan")
    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    @patch("setup.MODEL_DIR", "stan")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_cleanup_non_critical_failure(
        self,
        mock_remove,
        mock_exists,
        mock_repackage,
        mock_prune,
        mock_copy,
        mock_install_deps,
    ):
        """Test that cleanup failures don't cause the entire build to fail."""
        mock_repackage.return_value = True
        mock_exists.return_value = True
        mock_remove.side_effect = OSError("Cannot remove file")
        
        mock_model = Mock()
        mock_model.exe_file = "/tmp/prophet.exe"
        
        with patch("setup.cmdstanpy.CmdStanModel", return_value=mock_model):
            with tempfile.TemporaryDirectory() as target_dir:
                # Should not raise exception even though cleanup fails
                setup.build_cmdstan_model(target_dir)


class TestPruneCmdStan:
    """Test suite for prune_cmdstan function."""

    def test_prune_cmdstan_structure(self):
        """Test that prune_cmdstan correctly restructures the cmdstan directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a mock cmdstan directory structure
            cmdstan_dir = Path(tmp_dir) / "cmdstan-2.33.1"
            cmdstan_dir.mkdir()
            
            # Create bin directory with executables
            bin_dir = cmdstan_dir / "bin"
            bin_dir.mkdir()
            (bin_dir / "diagnose").touch()
            (bin_dir / "print").touch()
            (bin_dir / "stanc").touch()
            (bin_dir / "stansummary").touch()
            (bin_dir / "unwanted_file").touch()
            
            # Create a subdirectory that should be removed
            (bin_dir / "subdir").mkdir()
            (bin_dir / "subdir" / "file.txt").touch()
            
            # Create tbb directories
            tbb_parent = cmdstan_dir / "stan" / "lib" / "stan_math" / "lib"
            tbb_parent.mkdir(parents=True)
            for tbb_dir in ["tbb", "tbb_2020.3"]:
                (tbb_parent / tbb_dir).mkdir()
                (tbb_parent / tbb_dir / "lib.so").touch()
            
            # Create other directories that should be removed
            (cmdstan_dir / "src").mkdir()
            (cmdstan_dir / "examples").mkdir()
            
            # Run prune
            setup.prune_cmdstan(str(cmdstan_dir))
            
            # Verify structure
            assert cmdstan_dir.exists()
            assert (cmdstan_dir / "bin").exists()
            
            # Check that only allowed binaries remain
            bin_files = list((cmdstan_dir / "bin").iterdir())
            bin_names = [f.name for f in bin_files if f.is_file()]
            assert "diagnose" in bin_names
            assert "print" in bin_names
            assert "stanc" in bin_names
            assert "stansummary" in bin_names
            assert "unwanted_file" not in bin_names
            
            # Check subdirectories in bin are removed
            assert not (cmdstan_dir / "bin" / "subdir").exists()
            
            # Check tbb directories exist
            for tbb_dir in ["tbb", "tbb_2020.3"]:
                assert (cmdstan_dir / "stan" / "lib" / "stan_math" / "lib" / tbb_dir).exists()
            
            # Check other directories were removed
            assert not (cmdstan_dir / "src").exists()
            assert not (cmdstan_dir / "examples").exists()

    def test_prune_cmdstan_missing_directory(self):
        """Test prune_cmdstan handles missing directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent = Path(tmp_dir) / "non_existent"
            
            # Should raise an error or handle gracefully
            with pytest.raises(Exception):
                setup.prune_cmdstan(str(non_existent))


class TestInstallCmdStanDeps:
    """Test suite for install_cmdstan_deps function."""

    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    @patch("setup.rmtree")
    @patch("os.path.isdir")
    def test_install_cmdstan_deps_repackage(
        self,
        mock_isdir,
        mock_rmtree,
        mock_repackage,
    ):
        """Test install_cmdstan_deps when repackaging is enabled."""
        mock_repackage.return_value = True
        mock_isdir.return_value = True
        
        with patch("setup.cmdstanpy.install_cmdstan", return_value=True) as mock_install:
            with tempfile.TemporaryDirectory() as tmp_dir:
                cmdstan_dir = Path(tmp_dir) / "cmdstan-2.33.1"
                setup.install_cmdstan_deps(cmdstan_dir)
                
                # Verify install_cmdstan was called
                assert mock_install.called

    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", False)
    def test_install_cmdstan_deps_failure(
        self,
        mock_repackage,
    ):
        """Test install_cmdstan_deps raises error on installation failure."""
        mock_repackage.return_value = True
        
        with patch("setup.cmdstanpy.install_cmdstan", return_value=False):
            with tempfile.TemporaryDirectory() as tmp_dir:
                cmdstan_dir = Path(tmp_dir) / "cmdstan-2.33.1"
                
                with pytest.raises(RuntimeError, match="CmdStan failed to install"):
                    setup.install_cmdstan_deps(cmdstan_dir)

    @patch("setup.repackage_cmdstan")
    @patch("setup.IS_WINDOWS", True)
    @patch("setup.maybe_install_cmdstan_toolchain")
    @patch("os.path.isdir")
    @patch("setup.rmtree")
    def test_install_cmdstan_deps_windows(
        self,
        mock_rmtree,
        mock_isdir,
        mock_toolchain,
        mock_repackage,
    ):
        """Test install_cmdstan_deps installs toolchain on Windows."""
        mock_repackage.return_value = True
        mock_isdir.return_value = False
        mock_toolchain.return_value = True
        
        with patch("setup.cmdstanpy.install_cmdstan", return_value=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                cmdstan_dir = Path(tmp_dir) / "cmdstan-2.33.1"
                setup.install_cmdstan_deps(cmdstan_dir)
                
                # Verify toolchain installation was attempted
                assert mock_toolchain.called


class TestMaybeInstallCmdStanToolchain:
    """Test suite for maybe_install_cmdstan_toolchain function."""

    @patch("setup.cmdstanpy.utils.cxx_toolchain_path")
    def test_toolchain_already_installed(self, mock_toolchain_path):
        """Test when toolchain is already installed."""
        mock_toolchain_path.return_value = "/path/to/toolchain"
        
        result = setup.maybe_install_cmdstan_toolchain()
        
        # Should return False since toolchain already exists
        assert result is False

    @patch("setup.cmdstanpy.utils.cxx_toolchain_path")
    def test_toolchain_installation_success(self, mock_toolchain_path):
        """Test successful toolchain installation."""
        # First call raises exception (not installed), second call succeeds
        mock_toolchain_path.side_effect = [Exception("Not found"), "/path/to/toolchain"]
        
        with patch("setup.cmdstanpy.install_cxx_toolchain.run_rtools_install") as mock_install:
            result = setup.maybe_install_cmdstan_toolchain()
            
            # Should return True since toolchain was installed
            assert result is True
            assert mock_install.called

    @patch("setup.cmdstanpy.utils.cxx_toolchain_path")
    def test_toolchain_installation_legacy(self, mock_toolchain_path):
        """Test toolchain installation with legacy cmdstanpy."""
        mock_toolchain_path.side_effect = [Exception("Not found"), "/path/to/toolchain"]
        
        # Simulate ImportError for newer import, use legacy
        with patch("setup.cmdstanpy.install_cxx_toolchain.run_rtools_install", side_effect=ImportError):
            with patch("setup.cmdstanpy.install_cxx_toolchain.main") as mock_main:
                result = setup.maybe_install_cmdstan_toolchain()
                
                assert result is True
                assert mock_main.called


class TestHelperFunctions:
    """Test suite for helper functions."""

    @patch.dict(os.environ, {"PROPHET_REPACKAGE_CMDSTAN": "true"})
    def test_repackage_cmdstan_true(self):
        """Test repackage_cmdstan returns True when env var is set."""
        assert setup.repackage_cmdstan() is True

    @patch.dict(os.environ, {"PROPHET_REPACKAGE_CMDSTAN": "false"})
    def test_repackage_cmdstan_false(self):
        """Test repackage_cmdstan returns False when env var is false."""
        assert setup.repackage_cmdstan() is False

    @patch.dict(os.environ, {"PROPHET_REPACKAGE_CMDSTAN": "0"})
    def test_repackage_cmdstan_zero(self):
        """Test repackage_cmdstan returns False when env var is 0."""
        assert setup.repackage_cmdstan() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_repackage_cmdstan_default(self):
        """Test repackage_cmdstan returns True by default."""
        assert setup.repackage_cmdstan() is True

    @patch.dict(os.environ, {"STAN_BACKEND": "CMDSTANPY"})
    def test_get_backends_from_env_single(self):
        """Test get_backends_from_env with single backend."""
        backends = setup.get_backends_from_env()
        assert backends == ["CMDSTANPY"]

    @patch.dict(os.environ, {"STAN_BACKEND": "CMDSTANPY,PYSTAN"})
    def test_get_backends_from_env_multiple(self):
        """Test get_backends_from_env with multiple backends."""
        backends = setup.get_backends_from_env()
        assert "CMDSTANPY" in backends
        assert "PYSTAN" in backends

    @patch.dict(os.environ, {}, clear=True)
    def test_get_backends_from_env_default(self):
        """Test get_backends_from_env with default value."""
        backends = setup.get_backends_from_env()
        assert backends == ["CMDSTANPY"]


class TestBuildModels:
    """Test suite for build_models function."""

    @patch("setup.build_cmdstan_model")
    @patch("setup.get_backends_from_env")
    def test_build_models_cmdstanpy(self, mock_get_backends, mock_build_cmdstan):
        """Test build_models with CMDSTANPY backend."""
        mock_get_backends.return_value = ["CMDSTANPY"]
        
        with tempfile.TemporaryDirectory() as target_dir:
            setup.build_models(target_dir)
            
            # Verify cmdstan model build was called
            assert mock_build_cmdstan.called

    @patch("setup.build_cmdstan_model")
    @patch("setup.get_backends_from_env")
    def test_build_models_pystan_raises_error(self, mock_get_backends, mock_build_cmdstan):
        """Test build_models raises error for PYSTAN backend."""
        mock_get_backends.return_value = ["PYSTAN"]
        
        with tempfile.TemporaryDirectory() as target_dir:
            with pytest.raises(ValueError, match="PyStan backend is not supported"):
                setup.build_models(target_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
