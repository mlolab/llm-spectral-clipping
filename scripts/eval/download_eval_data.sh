#!/bin/bash
# Downloads all benchmark datasets that lm-eval needs into a shared HF cache.
#
# Usage:
#   bash scripts/eval/download_eval_data.sh

set -e

source your_path_uv/bin/activate

# Set cache directory to shared filesystem
export HF_HOME=your_path_hf_cache

echo "Downloading benchmark datasets to $HF_HOME ..."

python -c "
from datasets import load_dataset

datasets_to_download = [
    ('Rowan/hellaswag', None),
    ('allenai/ai2_arc', 'ARC-Easy'),
    ('allenai/ai2_arc', 'ARC-Challenge'),
    ('baber/piqa', None),
    ('allenai/winogrande', 'winogrande_xl'),
    ('aps/super_glue', 'boolq'),
    ('EleutherAI/lambada_openai', 'default'),
    ('allenai/openbookqa', 'main'),
    ('allenai/sciq', None),
]

for name, config in datasets_to_download:
    desc = f'{name} ({config})' if config else name
    print(f'Downloading {desc} ...')
    try:
        load_dataset(name, config)
        print(f'  OK')
    except Exception as e:
        print(f'  WARNING: {e}')

# MMLU: each of the 57 subjects needs its own config
mmlu_subjects = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology',
    'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies',
    'machine_learning', 'management', 'marketing', 'medical_genetics',
    'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology',
    'us_foreign_policy', 'virology', 'world_religions',
]

print(f'Downloading MMLU ({len(mmlu_subjects)} subjects) ...')
for subject in mmlu_subjects:
    try:
        load_dataset('cais/mmlu', subject)
        print(f'  {subject}: OK')
    except Exception as e:
        print(f'  {subject}: WARNING: {e}')

print('All datasets downloaded.')
"

echo ""
echo "Done. On compute nodes, set:"
echo "  export HF_HOME=your_path_hf_cache"
echo "  export HF_DATASETS_OFFLINE=1"
echo "  export TRANSFORMERS_OFFLINE=1"
