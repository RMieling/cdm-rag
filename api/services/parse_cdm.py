import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

import tqdm
from cdm.objectmodel import CdmCorpusDefinition, CdmEntityDefinition, CdmManifestDefinition
from cdm.storage import LocalAdapter
from cdm.utilities import AttributeResolutionDirectiveSet, ResolveOptions
from pyprojroot import here

from api.utils.logger import parse_cdm_logger


# Setup Logging
class SuppressGuidanceWarning(logging.Filter):
    def filter(self, record):
        if "Resolution guidance is being deprecated" in record.getMessage():
            return False
        return True


logging.getLogger("cdm").addFilter(SuppressGuidanceWarning())
logging.getLogger("cdm-python").addFilter(SuppressGuidanceWarning())

# Fix JSON Path leaks here for now
_original_default = json.JSONEncoder.default


def _patched_default(self, obj):
    if isinstance(obj, Path):
        return str(obj)
    return _original_default(self, obj)


json.JSONEncoder.default = _patched_default  # type: ignore[method-assign]


# Setup CDM corpus
def setup_cdm_corpus():
    cdm_base_dir = here() / "data" / "CDM"
    cdm_schema_dir = cdm_base_dir / "schemaDocuments"
    output_schemas_dir = here() / "data" / "output_schemas"
    os.makedirs(output_schemas_dir, exist_ok=True)

    corpus = CdmCorpusDefinition()
    local_adapter = LocalAdapter(root=cdm_schema_dir)
    corpus.storage.mount("local", local_adapter)
    corpus.storage.mount("cdm", local_adapter)
    corpus.storage.mount("output", LocalAdapter(root=output_schemas_dir))
    corpus.storage.default_namespace = "local"

    return corpus, output_schemas_dir


# Main traversal function
async def fetch_and_traverse_manifest(
    manifest_name: str,
    load_cached_resolved: bool = False,
    parent_manifest: CdmManifestDefinition = None,
    traverse_submanifests: bool = True,
    show_progress: bool = False,
    on_node_resolved: Optional[Callable[[CdmEntityDefinition], None]] = None,
    on_manifest_parsed: Optional[Callable[[CdmManifestDefinition], None]] = None,
):
    """
    Function that recursively traverses through the manifests
    and resolves the entities in each of them through the CDM SDK
    """

    corpus, _ = setup_cdm_corpus()
    manifest_path = f"local:{str(manifest_name)}"
    manifest_resolved_name = manifest_name.split("/")[-1].replace(
        ".manifest.cdm.json", "_with_relationships.manifest.cdm.json"
    )
    manifest_output_path = "output:" + "/".join(manifest_name.split("/")[:-1]) + "/" + manifest_resolved_name

    if load_cached_resolved:
        try:
            # raw_manifest = await corpus.fetch_object_async(manifest_path, relative_object=parent_manifest)
            # manifest = await corpus.fetch_object_async(str(manifest_output_path), relative_object=raw_manifest)
            manifest = await corpus.fetch_object_async(str(manifest_output_path))
        except Exception as e:
            parse_cdm_logger.error(f"Failed to load cached manifest: {e}")
            return None
    else:
        manifest = await corpus.fetch_object_async(manifest_path, relative_object=parent_manifest)
        if not manifest:
            parse_cdm_logger.error(f"Failed to fetch manifest: {manifest_path}")
            return None

        await corpus.calculate_entity_graph_async(manifest)
        await manifest.save_as_async(manifest_output_path, save_referenced=False)

    parse_cdm_logger.info(f"Found {len(manifest.entities)} entities in {manifest.name}. Processing...")

    # Resolve entities in manifest
    for entity in tqdm.tqdm(manifest.entities, disable=not show_progress):
        entity_path = f"local:{manifest.folder_path + entity.entity_path}"
        resolved_entity_name = f"{entity.entity_name}_Resolved.cdm.json"
        entity_output_path = manifest_output_path.replace(".manifest.cdm.json", f"_entities/{resolved_entity_name}")

        resolved_entity = None
        if load_cached_resolved:
            try:
                cached_obj = await corpus.fetch_object_async(entity_output_path)
                resolved_entity = cached_obj.definitions[0]
            except Exception as e:
                parse_cdm_logger.error(f"Failed to load cached resolved entity {entity_output_path}: {e} ")
        else:
            wrt_entity = await corpus.fetch_object_async(entity_path)
            res_opt = ResolveOptions(wrt_entity, AttributeResolutionDirectiveSet({"normalized", "referenceOnly"}))
            resolved_entity = await wrt_entity.create_resolved_entity_async(
                resolved_entity_name.replace(".cdm.json", ""), res_opt
            )
            await resolved_entity.in_document.save_as_async(entity_output_path, save_referenced=False)

        # Optional callback to ingest entities into the GraphManager
        if resolved_entity and on_node_resolved:
            on_node_resolved(resolved_entity)

    # Process the relationships in the manifest
    parse_cdm_logger.info(f"Found {len(manifest.relationships)} relationships in {manifest.name}.")

    # Optional callback to ingest relationships into the GraphManager
    if on_manifest_parsed:
        on_manifest_parsed(manifest)

    # RECURSE INTO SUB-MANIFESTS
    if traverse_submanifests:
        for sub_manifest in manifest.sub_manifests:
            parse_cdm_logger.info(f"Traversing into sub-manifest: {sub_manifest.definition}")
            await fetch_and_traverse_manifest(
                manifest_name=manifest.folder_path + sub_manifest.definition,
                load_cached_resolved=load_cached_resolved,
                parent_manifest=manifest,
                traverse_submanifests=traverse_submanifests,
                show_progress=show_progress,
                on_node_resolved=on_node_resolved,
                on_manifest_parsed=on_manifest_parsed,
            )

    return manifest


def check_or_retrieve_cdm_repo():
    repo_url = "git@github.com:microsoft/CDM.git"
    cdm_base_dir = here() / "data" / "CDM"

    if os.path.exists(cdm_base_dir) and os.path.isdir(cdm_base_dir):
        parse_cdm_logger.info(f"CDM Data folder already exists in {cdm_base_dir}, skipping download.")
    else:
        parse_cdm_logger.info(f"Cloning {repo_url} into {cdm_base_dir}...")
        try:
            subprocess.run(["git", "clone", "--", repo_url, cdm_base_dir], check=True)  # noqa: S603, S607
            parse_cdm_logger.info("Download complete.")
        except subprocess.CalledProcessError as e:
            parse_cdm_logger.error(f"Git clone failed: {e}")


async def ensure_manifest_resolved(traverse_manifest_files: dict):
    """
    Check if the _with_relationships files exist in output_schemas.
    If not, checks if CDM repo is downloaded and triggers the CDM SDK to generate them.
    """
    _, output_schemas_dir = setup_cdm_corpus()
    output_base_path = Path(output_schemas_dir)

    for manifest_path, traverse_submanifests in traverse_manifest_files.items():
        # Clean the string to prevent absolute path joining issues
        clean_path = manifest_path.lstrip("/")

        # Calculate expected output path structure
        path_parts = clean_path.split("/")
        folder_path = "/".join(path_parts[:-1])
        original_filename = path_parts[-1]
        resolved_filename = original_filename.replace(".manifest.cdm.json", "_with_relationships.manifest.cdm.json")

        # e.g., data/output_schemas/core/applicationCommon/applicationCommon_with_relationships.manifest.cdm.json
        expected_file = output_base_path / folder_path / resolved_filename

        if expected_file.exists():
            continue

        parse_cdm_logger.info(
            f"Resolved schemas not found for {original_filename}. Generating now (this may take a moment)..."
        )
        try:
            check_or_retrieve_cdm_repo()

            await fetch_and_traverse_manifest(
                manifest_name=manifest_path,
                load_cached_resolved=False,  # Force generation
                parent_manifest=None,
                traverse_submanifests=traverse_submanifests,
                show_progress=True,  # Turn on tqdm so you can see it working during startup
            )
        except Exception as e:
            parse_cdm_logger.error(f"An error occurred while generating schemas for {manifest_path}: {e}")


if __name__ == "__main__":
    traverse_manifest_files = {
        # common models
        "/core/applicationCommon/applicationCommon.manifest.cdm.json": False,  # also includes foundationCommon manifest
        "/core/operationsCommon/Entities/Common/Common.manifest.cdm.json": True,
        # finance objects
        "/core/operationsCommon/Entities/Finance/Finance.manifest.cdm.json": True,
        "/FinancialServices/FinancialServices.manifest.cdm.json": True,
    }

    asyncio.run(ensure_manifest_resolved(traverse_manifest_files))

    manifest_path, traverse_submanifests = next(iter(traverse_manifest_files.items()))
    asyncio.run(
        fetch_and_traverse_manifest(
            manifest_name=manifest_path,
            load_cached_resolved=True,  # Force generation
            parent_manifest=None,
            traverse_submanifests=traverse_submanifests,
            show_progress=True,  # Turn on tqdm so you can see it working during startup
        )
    )
