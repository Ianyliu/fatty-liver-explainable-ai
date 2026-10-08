"""Compose the three original README result images without redrawing their data.

The original PNGs remain private cached sources. Their historical uncertainty and
classifier estimates must not be represented as newly validated results.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from proceedings_data import ROOT, hash_entry, require
from proceedings_style import configure, save


def original_results(output):
    manifest = json.loads((ROOT / 'manuscript/proceedings_2026/assets/readme_result_references.json').read_text())
    rows = manifest['references']
    require([row['method'] for row in rows] == ['marginal', 'elastic_net', 'ridge'],
            'Original result panels must preserve README order')
    configure()
    width = 5.5
    margin = .07
    gap = .25
    images = []
    heights = []
    sources = []
    for row in rows:
        path = ROOT / row['path']
        source = hash_entry(path)
        require(source['sha256'] == row['sha256'], 'Original README image hash differs: ' + str(path))
        image = plt.imread(path)
        images.append(image)
        heights.append((width - 2 * margin) * image.shape[0] / image.shape[1])
        sources.append(source)
    height = sum(heights) + 3 * gap + margin
    fig = plt.figure(figsize=(width, height))
    top = height
    for image, panel_height, letter, title in zip(
            images, heights, 'ABC', ['Marginal influence: Pearson correlation',
                                    'Conditional influence: Elastic Net',
                                    'Conditional influence: Ridge']):
        fig.text(margin / width, (top - .16) / height, letter + '  ' + title,
                 fontsize=9, weight='bold')
        top -= gap
        ax = fig.add_axes([margin / width, (top - panel_height) / height,
                           (width - 2 * margin) / width, panel_height / height])
        ax.imshow(image, interpolation='none')
        ax.axis('off')
        top -= panel_height
    save(fig, Path(output), 'figure2_influence')
    caption = (
        'Composite of the three original README result illustrations, reproduced without redrawing: '
        'A, marginal Pearson correlation; B, conditional Elastic Net; C, conditional Ridge. '
        'Ultrasound thumbnails, original image labels, bar ordering, colors, error bars and faded bars '
        'are preserved from the historical artwork. These panels illustrate the original explanation '
        'presentation; their classifier coefficients, uncertainty intervals and significance coding '
        'have not been independently revalidated and are not quantitative evidence for the current '
        'probability-regression experiments. Current estimates are reported separately in the tables '
        'and evaluation figures, with a validated patient-level example in the appendix.')
    return caption, sources
