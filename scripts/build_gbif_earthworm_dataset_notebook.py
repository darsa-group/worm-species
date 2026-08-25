#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/gbif_earthworm_dataset_audit.ipynb")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown("""# GBIF true-earthworm image dataset audit

This notebook audits the user-approved earthworm scope (`Crassiclitellata` and
`Moniligastrida`) after requiring a GBIF genus key. It uses every distinct
still image attached to each occurrence. White worms, aquatic oligochaetes,
branchiobdellids, and leeches are outside scope.

The notebook prefers the curated manifest, then the downloaded manifest, then
the raw media manifest. GBIF identifications are occurrence metadata, not
independently verified image labels. Re-run all cells after each acquisition or
curation update; do not interpret an unexecuted notebook as dataset results.
"""),
    code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import yaml

CONFIG_PATH = Path('../configs/gbif_oligochaeta.yaml')
with CONFIG_PATH.open() as handle:
    config = yaml.safe_load(handle)

workspace = {key: Path('..') / value for key, value in config['workspace'].items()}
candidates = [workspace['curated_manifest'], workspace['downloaded_manifest'], workspace['manifest']]
MANIFEST_PATH = next((path for path in candidates if path.is_file()), None)
if MANIFEST_PATH is None:
    raise FileNotFoundError('No GBIF manifest exists yet. Run acquisition/build-manifest first.')

df = pd.read_csv(MANIFEST_PATH, low_memory=False)
print(f'Manifest: {MANIFEST_PATH}')
print(f'Rows (distinct occurrence-image pairs): {len(df):,}')
df.head()
"""),
    markdown("## 1. Scope, completeness, and label coverage"),
    code("""def nunique(column):
    return int(df[column].nunique(dropna=True)) if column in df else None

overview = pd.Series({
    'image rows': len(df),
    'GBIF occurrences': nunique('gbif_id'),
    'orders': nunique('order_key'),
    'families': nunique('family_key'),
    'genera': nunique('genus_key'),
    'species-labelled rows': int(df.get('species_key', pd.Series(dtype=object)).notna().sum()),
    'unique species keys': nunique('species_key'),
    'datasets': nunique('dataset_key'),
})
display(overview.to_frame('value'))

assert df['genus'].notna().all() and df['genus_key'].notna().all(), 'Genus-defined contract violated'
expected_orders = {str(item['key']) for item in config['gbif']['included_orders']}
observed_orders = set(df['order_key'].dropna().astype(str))
assert observed_orders <= expected_orders, f'Out-of-scope order keys: {observed_orders - expected_orders}'

if 'download_status' in df:
    display(df['download_status'].value_counts(dropna=False).to_frame('images'))
    per_occurrence_status = df.groupby('gbif_id')['download_status'].agg(
        images='size', downloaded=lambda values: (values == 'downloaded').sum())
    per_occurrence_status['complete'] = per_occurrence_status['images'] == per_occurrence_status['downloaded']
    display(per_occurrence_status['complete'].value_counts().rename(index={True: 'complete', False: 'incomplete'}).to_frame('occurrences'))
"""),
    markdown("## 2. Multiple images per occurrence"),
    code("""images_per_occurrence = df.groupby('gbif_id').size().rename('images')
display(images_per_occurrence.describe(percentiles=[.5, .75, .9, .95, .99]).to_frame())
ax = images_per_occurrence.clip(upper=images_per_occurrence.quantile(.99)).plot.hist(
    bins=40, figsize=(9, 4), color='#4477AA', edgecolor='white')
ax.set(title='Images per GBIF occurrence (clipped at 99th percentile)', xlabel='Images', ylabel='Occurrences')
plt.show()
display(images_per_occurrence.sort_values(ascending=False).head(20).to_frame())
"""),
    markdown("## 3. Taxonomic distributions"),
    code("""def top_distribution(column, n=30):
    counts = df[column].fillna('<missing>').value_counts().head(n).sort_values()
    ax = counts.plot.barh(figsize=(9, max(3, n * .22)), color='#228833')
    ax.set(title=f'Top {column} values by image count', xlabel='Images', ylabel=column)
    plt.tight_layout(); plt.show()
    return counts.sort_values(ascending=False).to_frame('images')

display(top_distribution('genus', 30))
display(top_distribution('species', 30))
display(top_distribution('family', 20))
"""),
    markdown("## 4. Geographic and temporal coverage"),
    code("""if 'country' in df:
    display(top_distribution('country', 30))

if {'decimal_latitude', 'decimal_longitude'} <= set(df):
    geo = df.drop_duplicates('gbif_id').copy()
    geo['decimal_latitude'] = pd.to_numeric(geo['decimal_latitude'], errors='coerce')
    geo['decimal_longitude'] = pd.to_numeric(geo['decimal_longitude'], errors='coerce')
    geo = geo.dropna(subset=['decimal_latitude', 'decimal_longitude'])
    print(f'Georeferenced occurrences: {len(geo):,}')
    if len(geo):
        fig = px.scatter_geo(geo, lat='decimal_latitude', lon='decimal_longitude',
                             color='genus', hover_name='scientific_name', opacity=.45,
                             title='Georeferenced GBIF earthworm occurrences')
        fig.show()

if 'year' in df:
    years = pd.to_numeric(df.drop_duplicates('gbif_id')['year'], errors='coerce').dropna()
    years = years[(years >= 1700) & (years <= pd.Timestamp.now().year)]
    ax = years.plot.hist(bins=40, figsize=(9, 4), color='#CC6677', edgecolor='white')
    ax.set(title='Occurrence year distribution', xlabel='Year', ylabel='Occurrences'); plt.show()
"""),
    markdown("## 5. Data sources, record types, and licences"),
    code("""for column in ['dataset_name', 'publisher', 'basis_of_record', 'license']:
    if column in df:
        display(top_distribution(column, 25))
"""),
    markdown("## 6. Image files, dimensions, and duplicates"),
    code("""for column in ['bytes', 'width', 'height']:
    if column in df:
        df[column] = pd.to_numeric(df[column], errors='coerce')
if {'width', 'height'} <= set(df):
    display(df[['width', 'height', 'bytes']].describe(percentiles=[.01, .1, .5, .9, .99]))
    fig = px.scatter(df.dropna(subset=['width', 'height']), x='width', y='height',
                     color='curation_label' if 'curation_label' in df else None,
                     log_x=True, log_y=True, opacity=.35, title='Downloaded image dimensions')
    fig.show()
if 'sha256' in df:
    valid_hash = df['sha256'].dropna().astype(str)
    duplicates = valid_hash.value_counts()
    print(f'Exact duplicate hashes: {(duplicates > 1).sum():,}; duplicate image rows: {(duplicates - 1).clip(lower=0).sum():,}')
    display(df[df['sha256'].isin(duplicates[duplicates > 1].index)].sort_values('sha256').head(100))
"""),
    markdown("## 7. DINOv3 clusters, UMAP projection, and curation"),
    code("""if {'projection_x', 'projection_y', 'cluster'} <= set(df):
    color = 'curation_label' if 'curation_label' in df else 'cluster'
    fig = px.scatter(df, x='projection_x', y='projection_y', color=color,
                     hover_data=['image_id', 'genus', 'species', 'cluster'],
                     render_mode='webgl', opacity=.55,
                     title='DINOv3 embeddings projected with UMAP')
    fig.show()
    display(pd.crosstab(df['cluster'], df.get('curation_label', 'unreviewed'), margins=True))
else:
    print('Cluster columns are not present yet; run embed, cluster, and curation stages.')
"""),
    markdown("## 8. Existing-model scope and predictions"),
    code("""scope_columns = [column for column in df if column.startswith('checkpoint_')]
if scope_columns:
    for column in scope_columns:
        display(df[column].value_counts(dropna=False).to_frame('images'))
if 'genus_label_agreement' in df:
    known = df['checkpoint_genus_scope'] == 'known'
    print('Known-genus GBIF-label agreement:', df.loc[known, 'genus_label_agreement'].mean())
    print('This is agreement with GBIF metadata, not independently verified accuracy.')
else:
    print('Existing-checkpoint predictions are not merged into this manifest yet.')
"""),
    markdown("## 9. Machine-readable audit summary"),
    code("""audit_summary = {
    'manifest': str(MANIFEST_PATH),
    'image_rows': int(len(df)),
    'occurrences': int(df['gbif_id'].nunique()),
    'genera': int(df['genus_key'].nunique()),
    'species_keys': int(df['species_key'].nunique()) if 'species_key' in df else None,
    'images_per_occurrence': images_per_occurrence.describe().to_dict(),
    'download_status': df['download_status'].value_counts(dropna=False).to_dict() if 'download_status' in df else None,
    'curation_status': df['curation_label'].value_counts(dropna=False).to_dict() if 'curation_label' in df else None,
}
display(audit_summary)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "wormspecies", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)

