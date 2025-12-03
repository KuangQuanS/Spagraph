#!/usr/bin/env python
"""Test script to verify scmapst package installation

Run this script after installing the package to verify everything works.
"""

import sys
import importlib


def test_import():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        import scmapst
        print(f"✓ scmapst imported successfully")
        print(f"  Version: {scmapst.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import scmapst: {e}")
        return False
    
    return True


def test_api_functions():
    """Test API function availability"""
    print("\nTesting API functions...")
    
    try:
        import scmapst
        
        # Check main API functions
        functions = ['train', 'deconvolve', 'analyze_cellchat']
        for func_name in functions:
            if hasattr(scmapst, func_name):
                print(f"✓ scmapst.{func_name} available")
            else:
                print(f"✗ scmapst.{func_name} not found")
                return False
    except Exception as e:
        print(f"✗ Error checking API functions: {e}")
        return False
    
    return True


def test_submodules():
    """Test submodule imports"""
    print("\nTesting submodules...")
    
    submodules = [
        'scmapst.training',
        'scmapst.training.stage1',
        'scmapst.training.stage2',
        'scmapst.training.stage3',
        'scmapst.cli'
    ]
    
    for module_name in submodules:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name} imported")
        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {e}")
            return False
    
    return True


def test_dependencies():
    """Test critical dependencies"""
    print("\nTesting dependencies...")
    
    dependencies = [
        'torch',
        'torch_geometric',
        'scanpy',
        'anndata',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} missing")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -e .")
        return False
    
    return True


def test_cli_commands():
    """Test CLI command availability"""
    print("\nTesting CLI commands...")
    
    import subprocess
    
    commands = [
        ['scmapst-train', '--help'],
        ['scmapst-deconvolve', '--help']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ {cmd[0]} available")
            else:
                print(f"✗ {cmd[0]} failed with exit code {result.returncode}")
                return False
        except FileNotFoundError:
            print(f"✗ {cmd[0]} not found in PATH")
            return False
        except subprocess.TimeoutExpired:
            print(f"✗ {cmd[0]} timed out")
            return False
        except Exception as e:
            print(f"✗ Error testing {cmd[0]}: {e}")
            return False
    
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("SC-MAP-ST Package Installation Test")
    print("="*70)
    
    tests = [
        ("Basic imports", test_import),
        ("API functions", test_api_functions),
        ("Submodules", test_submodules),
        ("Dependencies", test_dependencies),
        ("CLI commands", test_cli_commands)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Test: {test_name}")
        print(f"{'='*70}")
        results.append(test_func())
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for (test_name, _), result in zip(tests, results):
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests passed! Package is ready to use.")
        print("\nNext steps:")
        print("  1. Run example: python example_usage.py")
        print("  2. Import in Python: import scmapst")
        print("  3. Use CLI: scmapst-train --help")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Make sure you installed the package: pip install -e .")
        print("  2. Check Python version: python --version (requires >=3.8)")
        print("  3. Verify virtual environment is activated")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
