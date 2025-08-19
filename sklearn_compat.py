"""
Scikit-learn compatibility module to handle pickle loading issues
when loading models saved with older sklearn versions.
"""

import sys
import sklearn
from sklearn import linear_model

# Handle the sklearn.linear_model.base module that was restructured in newer versions
if not hasattr(linear_model, 'base'):
    # Create a compatibility module for sklearn.linear_model.base
    import types
    base_module = types.ModuleType('sklearn.linear_model.base')
    
    # Import the classes that were previously in sklearn.linear_model.base
    try:
        from sklearn.linear_model._base import LinearRegression as _LinearRegression
        from sklearn.linear_model._base import LinearModel, RegressorMixin
        base_module.LinearRegression = _LinearRegression
        base_module.LinearModel = LinearModel
        base_module.RegressorMixin = RegressorMixin
    except ImportError:
        # Fallback for even newer versions
        from sklearn.linear_model import LinearRegression as _LinearRegression
        from sklearn.base import BaseEstimator, RegressorMixin
        base_module.LinearRegression = _LinearRegression
        base_module.BaseEstimator = BaseEstimator
        base_module.RegressorMixin = RegressorMixin
    
    # Make it available as sklearn.linear_model.base
    linear_model.base = base_module
    sys.modules['sklearn.linear_model.base'] = base_module

# Additional compatibility fixes for other common sklearn pickle issues
try:
    import sklearn.cross_validation
except ImportError:
    # sklearn.cross_validation was renamed to sklearn.model_selection
    import sklearn.model_selection
    sys.modules['sklearn.cross_validation'] = sklearn.model_selection

try:
    import sklearn.grid_search
except ImportError:
    # sklearn.grid_search was moved to sklearn.model_selection
    import sklearn.model_selection
    sys.modules['sklearn.grid_search'] = sklearn.model_selection

# Handle sklearn.externals.joblib which was removed in newer versions
try:
    import sklearn.externals.joblib
except ImportError:
    import joblib
    import types
    externals_module = types.ModuleType('sklearn.externals')
    externals_module.joblib = joblib
    sklearn.externals = externals_module
    sys.modules['sklearn.externals'] = externals_module
    sys.modules['sklearn.externals.joblib'] = joblib

# Fix for missing 'positive' attribute in LinearRegression models
try:
    from sklearn.linear_model import LinearRegression
    
    # Monkey patch to add missing attributes that newer sklearn expects
    if not hasattr(LinearRegression, 'positive'):
        LinearRegression.positive = False
    
    # Also patch any existing instances that might be loaded from pickle
    def patch_linear_regression_instance(obj):
        if hasattr(obj, '__class__') and 'LinearRegression' in str(obj.__class__):
            if not hasattr(obj, 'positive'):
                obj.positive = False
        return obj
    
    # Store the original pickle load function
    import pickle
    _original_pickle_load = pickle.load
    
    def patched_pickle_load(file):
        obj = _original_pickle_load(file)
        # Try to patch the loaded object if it's a LinearRegression
        if hasattr(obj, '__class__') and 'LinearRegression' in str(obj.__class__):
            if not hasattr(obj, 'positive'):
                obj.positive = False
        # If it's a dict containing models, patch them too
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if hasattr(value, '__class__') and 'LinearRegression' in str(value.__class__):
                    if not hasattr(value, 'positive'):
                        value.positive = False
                elif isinstance(value, dict) and 'model' in value:
                    model = value['model']
                    if hasattr(model, '__class__') and 'LinearRegression' in str(model.__class__):
                        if not hasattr(model, 'positive'):
                            model.positive = False
        return obj
    
    # Replace pickle.load with our patched version
    pickle.load = patched_pickle_load
    
except ImportError:
    pass

print("Sklearn compatibility module loaded successfully")
