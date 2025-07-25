from typing import Dict, List, Tuple

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

@custom
def models(*args, **kwargs) -> Tuple[List[str], List[Dict[str, str]]]:
    model_names: str = kwargs.get(
        'models',
        'ensemble.RandomForestRegressor'
    )

    child_data: List[str] = [
        model_name.strip() for model_name in model_names.split(',')
    ]

    child_metadata: List[Dict[str, str]] = [
        dict(block_uuid=model_name.split('.')[-1]) for model_name in child_data
    ]

    return child_data, child_metadata
