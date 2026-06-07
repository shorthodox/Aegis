#!/usr/bin/env python3
"""
Unit tests for validation concatenation safety fix.
Verifies that validation handles empty folds and regimes gracefully.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from validation import _validate_fold_inputs


def test_validate_fold_inputs_empty_list():
    """Test: Empty fold list should be rejected."""
    print("\nTest 1: Empty fold list")
    
    check = _validate_fold_inputs([])
    
    assert check['status'] == 'validation_failed', "Should reject empty fold list"
    assert 'No folds provided' in check['reasons'][0], f"Wrong message: {check['reasons']}"
    assert check['can_proceed'] is False, "Should not allow proceed"
    
    print(f"  ✓ PASS: Correctly rejected empty list")
    print(f"    Message: {check['reasons'][0]}")


def test_validate_fold_inputs_single_fold():
    """Test: Single fold should be flagged as insufficient for LOFO."""
    print("\nTest 2: Single fold (LOFO not possible)")
    
    df = pd.DataFrame({'pnl': [0.01, -0.02, 0.05]})
    check = _validate_fold_inputs([df])
    
    assert check['status'] == 'validation_failed', "Should reject single fold"
    assert any('only 1 fold' in r.lower() for r in check['reasons']), f"Wrong message: {check['reasons']}"
    assert check['can_proceed'] is False, "Should not allow proceed"
    
    print(f"  ✓ PASS: Correctly flagged single fold")
    print(f"    Message: {check['reasons']}")


def test_validate_fold_inputs_all_empty():
    """Test: All empty DataFrames should be rejected."""
    print("\nTest 3: All folds empty (no trades)")
    
    df_empty1 = pd.DataFrame({'pnl': []})
    df_empty2 = pd.DataFrame({'pnl': []})
    
    check = _validate_fold_inputs([df_empty1, df_empty2])
    
    assert check['status'] == 'validation_failed', "Should reject all-empty folds"
    assert any('empty' in r.lower() for r in check['reasons']), f"Wrong message: {check['reasons']}"
    assert check['can_proceed'] is False, "Should not allow proceed"
    
    print(f"  ✓ PASS: Correctly rejected all-empty folds")
    print(f"    Message: {check['reasons']}")


def test_validate_fold_inputs_missing_columns():
    """Test: Missing required columns should be flagged."""
    print("\nTest 4: Missing required columns")
    
    df_bad = pd.DataFrame({'return': [0.01, -0.02]})  # Missing 'pnl' column
    df_good = pd.DataFrame({'pnl': [0.05, -0.01]})
    
    check = _validate_fold_inputs([df_bad, df_good])
    
    assert check['status'] == 'validation_failed', "Should reject folds with missing columns"
    assert any('missing columns' in r.lower() for r in check['reasons']), f"Wrong message: {check['reasons']}"
    assert check['can_proceed'] is False, "Should not allow proceed"
    
    print(f"  ✓ PASS: Correctly flagged missing columns")
    print(f"    Message: {check['reasons']}")


def test_validate_fold_inputs_non_dataframe():
    """Test: Non-DataFrame inputs should be rejected."""
    print("\nTest 5: Non-DataFrame input (e.g., list)")
    
    check = _validate_fold_inputs([{'pnl': [0.01, -0.02]}])  # Dict, not DataFrame
    
    assert check['status'] == 'validation_failed', "Should reject non-DataFrame"
    assert any('not a dataframe' in r.lower() for r in check['reasons']), f"Wrong message: {check['reasons']}"
    assert check['can_proceed'] is False, "Should not allow proceed"
    
    print(f"  ✓ PASS: Correctly rejected non-DataFrame")
    print(f"    Message: {check['reasons'][0]}")


def test_validate_fold_inputs_valid():
    """Test: Valid folds should pass."""
    print("\nTest 6: Valid folds (multiple with trades)")
    
    df1 = pd.DataFrame({'pnl': [0.01, -0.02, 0.05]})
    df2 = pd.DataFrame({'pnl': [0.03, -0.01, 0.02]})
    df3 = pd.DataFrame({'pnl': [0.04, -0.03, 0.01]})
    
    check = _validate_fold_inputs([df1, df2, df3])
    
    assert check['status'] == 'validation_passed', "Should accept valid folds"
    assert check['can_proceed'] is True, "Should allow proceed"
    assert len(check['reasons']) == 0, "Should have no issues"
    
    print(f"  ✓ PASS: Valid folds accepted")


def test_validate_fold_inputs_mixed_empty_nonempty():
    """Test: Mix of empty and non-empty folds should accept if enough trades."""
    print("\nTest 7: Mix of empty and non-empty folds")
    
    df_empty = pd.DataFrame({'pnl': []})
    df_trades = pd.DataFrame({'pnl': [0.01, -0.02, 0.05]})
    
    check = _validate_fold_inputs([df_empty, df_trades])
    
    # This should PASS because total trades > 0
    # (The function checks total trades, not individual fold sizes)
    assert check['status'] == 'validation_passed', "Should accept if total trades > 0"
    assert check['can_proceed'] is True, "Should allow proceed"
    
    print(f"  ✓ PASS: Mixed folds accepted")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TESTING VALIDATION FOLD INPUT DIAGNOSTIC FUNCTION")
    print("=" * 60)
    
    try:
        test_validate_fold_inputs_empty_list()
        test_validate_fold_inputs_single_fold()
        test_validate_fold_inputs_all_empty()
        test_validate_fold_inputs_missing_columns()
        test_validate_fold_inputs_non_dataframe()
        test_validate_fold_inputs_valid()
        test_validate_fold_inputs_mixed_empty_nonempty()
        
        print("\n" + "=" * 60)
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("=" * 60)
        print("\nConclusion: The _validate_fold_inputs() function correctly")
        print("identifies problematic inputs before concatenation, preventing")
        print("'No objects to concatenate' errors.")
        print("=" * 60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
