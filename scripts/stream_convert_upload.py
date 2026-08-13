#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from mlx_minimax_music3.conversion import (
    MANIFEST_NAME,
    convert_component_source_shard,
    mark_remote_verified,
    source_weight_inventory,
)
from mlx_minimax_music3.prepare import prepare_checkpoint_layout


def remote_file(api: HfApi, repo_id: str, path: str):
    info = api.model_info(repo_id, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == path:
            return info.sha, sibling
    return info.sha, None


def verify_and_mark(
    api: HfApi,
    repo_id: str,
    output_root: Path,
    component: str,
) -> dict:
    component_dir = output_root / component
    manifest_path = component_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        path_in_repo = f"{component}/{shard['file']}"
        local_path = component_dir / shard["file"]
        revision, remote = remote_file(api, repo_id, path_in_repo)
        remote_hash = getattr(getattr(remote, "lfs", None), "sha256", None) if remote else None
        if remote_hash != shard["sha256"]:
            if not local_path.is_file():
                raise FileNotFoundError(f"unuploaded output shard is missing locally: {local_path}")
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                commit_message=f"upload converted {component} shard {shard['file']}",
            )
            revision, remote = remote_file(api, repo_id, path_in_repo)
            remote_hash = getattr(getattr(remote, "lfs", None), "sha256", None) if remote else None
        if remote_hash != shard["sha256"]:
            raise RuntimeError(f"remote LFS SHA-256 verification failed: {path_in_repo}")
        manifest = mark_remote_verified(
            component_dir,
            shard["file"],
            remote_hash,
            revision,
        )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=manifest_path,
        path_in_repo=f"{component}/{MANIFEST_NAME}",
        commit_message=f"record verified {component} conversion progress",
    )
    if manifest["status"] == "complete":
        index_path = component_dir / "model.safetensors.index.json"
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=index_path,
            path_in_repo=f"{component}/model.safetensors.index.json",
            commit_message=f"complete {component} shard index",
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download, convert, upload, verify, and optionally clean one official source shard"
    )
    parser.add_argument("--source-repo", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--source-shard", required=True)
    parser.add_argument("--shard-size-gib", type=float, default=1.0)
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--delete-output", action="store_true")
    args = parser.parse_args()
    if args.shard_size_gib <= 0:
        parser.error("--shard-size-gib must be positive")

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    prepare_checkpoint_layout(source_root, output_root)
    inventory = source_weight_inventory(source_root / args.component)
    inventory_names = {path.name for path in inventory}
    if args.source_shard not in inventory_names:
        raise ValueError(f"source shard is not listed in the official index: {args.source_shard}")

    source_path = Path(
        hf_hub_download(
            repo_id=args.source_repo,
            filename=f"{args.component}/{args.source_shard}",
            revision=args.source_revision,
            local_dir=source_root,
        )
    )
    manifest = convert_component_source_shard(
        source_root,
        output_root,
        args.component,
        args.source_shard,
        shard_size=int(args.shard_size_gib * 1024**3),
    )
    api = HfApi()
    manifest = verify_and_mark(api, args.target_repo, output_root, args.component)
    if not all(shard.get("remote_verified") for shard in manifest["shards"]):
        raise RuntimeError("not every converted shard is remotely verified")

    if args.delete_output:
        for shard in manifest["shards"]:
            local = output_root / args.component / shard["file"]
            if local.exists() and shard.get("remote_verified"):
                local.unlink()
    if args.delete_source:
        source_path.unlink()

    print(
        json.dumps(
            {
                "component": args.component,
                "source_shard": args.source_shard,
                "status": manifest["status"],
                "processed_source_files": manifest["processed_source_files"],
                "remote_verified_shards": sum(
                    1 for shard in manifest["shards"] if shard.get("remote_verified")
                ),
                "source_deleted": args.delete_source,
                "output_deleted": args.delete_output,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
