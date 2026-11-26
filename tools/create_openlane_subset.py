import os
import re
import json
import numpy as np
import shutil
from collections import defaultdict, Counter


LOOK_CONDS = {
    'scene': [
        'residential',
        'urban',
        'suburbs',
        'highway',
        'parking lot'
    ],
    'weather': [
        'clear',
        'partly cloudy',
        'overcast',
        'rainy',
        'foggy'
    ],
    'hours': [
        'daytime',
        'night',
        'dawn/dusk'
    ]
}

ALIASES = {
    'scene': {
        'suburban': 'suburbs',
        'suburb': 'suburbs'
    }
}


def load_json(json_pth):

    if not os.path.exists(json_pth):
        raise FileNotFoundError(f"File not found at {json_pth}")

    with open(json_pth, 'r') as infile:
        return json.load(infile)


def get_condition(value, key, segment):

    value = str(value).lower().strip()
    value = re.sub(r'[_-]+', ' ', value)

    value = ALIASES.get(key, {}).get(value, value)

    if value in LOOK_CONDS[key]:
        return value

    matches = [
        category
        for category in LOOK_CONDS[key]
        if re.search(
            rf'(?<!\w){re.escape(category)}(?!\w)',
            value
        )
    ]

    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Cannot uniquely identify {key} condition for segment "
        f"'{segment}': '{value}', matches={matches}"
    )


def cond_based_filter(
    cond_data,
    split_dir,
    sub_dataset_size,
    seed=42
):

    buckets = defaultdict(list)

    cond_data = {
        str(segment).strip(): metadata
        for segment, metadata in cond_data.items()
    }

    for segment in sorted(os.listdir(split_dir)):

        segment_dir = os.path.join(split_dir, segment)

        if not os.path.isdir(segment_dir):
            continue

        segment = segment.strip()
        conditions = cond_data.get(segment)

        if conditions is None:
            print(f"Warning: No metadata for segment '{segment}'")
            continue

        scene = get_condition(
            conditions['scene'], 'scene', segment
        )

        weather = get_condition(
            conditions['weather'], 'weather', segment
        )

        hours = get_condition(
            conditions['hours'], 'hours', segment
        )

        combo = (scene, weather, hours)

        files = sorted(
            f for f in os.listdir(segment_dir)
            if f.lower().endswith('.json')
        )

        for filename in files:

            sample_id = f"{segment}/{filename}"

            buckets[combo].append(sample_id)

    total_available = sum(len(v) for v in buckets.values())

    if total_available < sub_dataset_size:
        raise RuntimeError(
            f"Only {total_available} eligible samples available, "
            f"but {sub_dataset_size} requested."
        )

    print(f"\nTotal available samples: {total_available}")

    for key_idx, key in enumerate(LOOK_CONDS):

        counts = Counter(
            combo[key_idx] for combo in buckets
            for _ in buckets[combo]
        )

        print(f"\nAvailable {key} samples:")

        for category in LOOK_CONDS[key]:
            print(f"  {category:20s}: {counts[category]}")

    rng = np.random.default_rng(seed)

    for samples in buckets.values():
        rng.shuffle(samples)

    quotas = {
        key: sub_dataset_size / len(LOOK_CONDS[key])
        for key in LOOK_CONDS
    }

    selected_counts = {
        key: Counter()
        for key in LOOK_CONDS
    }

    selected_samples = []

    while len(selected_samples) < sub_dataset_size:

        best_combo = None
        best_score = float('inf')

        for combo, samples in buckets.items():

            if not samples:
                continue

            score = 0.0

            for key, category in zip(LOOK_CONDS, combo):

                target = quotas[key]
                current = selected_counts[key][category]

                delta = (
                    (current + 1 - target) ** 2
                    - (current - target) ** 2
                ) / (target ** 2)

                score += delta

            score /= len(LOOK_CONDS)

            score += rng.uniform(0, 1e-10)

            if score < best_score:
                best_score = score
                best_combo = combo

        if best_combo is None:
            raise RuntimeError(
                "No remaining candidates before reaching "
                "the requested subset size."
            )

        sample_id = buckets[best_combo].pop()

        selected_samples.append(sample_id)

        for key, category in zip(LOOK_CONDS, best_combo):
            selected_counts[key][category] += 1

    print(f"\nSelected {len(selected_samples)} samples")

    for key in LOOK_CONDS:

        print(f"\nSelected {key} distribution:")

        for category in LOOK_CONDS[key]:

            count = selected_counts[key][category]
            percentage = 100 * count / sub_dataset_size

            print(
                f"  {category:20s}: {count:5d} "
                f"({percentage:.2f}%)"
            )

    return sorted(selected_samples)

def copy_selected_annotations(
    selected_samples,
    source_split_dir,
    output_split_dir
):
    if os.path.exists(output_split_dir):
        shutil.rmtree(output_split_dir)

    for sample_id in selected_samples:

        # sample_id = segment_name/annotation_filename.json
        relative_path = os.path.join(*sample_id.split('/'))

        src_path = os.path.join(source_split_dir, relative_path)
        dst_path = os.path.join(output_split_dir, relative_path)

        if not os.path.isfile(src_path):
            raise FileNotFoundError(
                f"Annotation not found: {src_path}"
            )

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        shutil.copy2(src_path, dst_path)

    print(
        f"Copied {len(selected_samples)} annotations "
        f"to {output_split_dir}"
    )

if __name__ == "__main__":

    root_dir = (
        r"C:\Users\oo_wa\Downloads\python_scripts"
        r"\BEVLaneDet2\data\OpenLane"
    )

    out_dir = os.path.join(root_dir, 'subset')
    os.makedirs(out_dir, exist_ok=True)

    cond_json_pth = os.path.join(root_dir, 'scene.json')
    cond_data = load_json(cond_json_pth)

    split_sizes = {
        'training': 6000,
        'validation': 1500
    }

    for split, subset_size in split_sizes.items():

        print(f"\nGenerating {split} subset...")

        source_split_dir = os.path.join(
            root_dir, 'lane3d_300', split
        )

        selected_samples = cond_based_filter(
            cond_data=cond_data,
            split_dir=source_split_dir,
            sub_dataset_size=subset_size,
            seed=42
        )

        output_split_dir = os.path.join(out_dir, split)

        copy_selected_annotations(
            selected_samples=selected_samples,
            source_split_dir=source_split_dir,
            output_split_dir=output_split_dir
        )