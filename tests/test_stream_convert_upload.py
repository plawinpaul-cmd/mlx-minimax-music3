import hashlib
import json
from types import SimpleNamespace

from scripts.stream_convert_upload import verify_and_mark


class FakeHubApi:
    def __init__(self, remote_hashes: dict[str, str] | None = None):
        self.remote_hashes = remote_hashes or {}
        self.uploads: list[dict] = []
        self.info_calls = 0

    def model_info(self, repo_id: str, *, files_metadata: bool):
        assert repo_id == "owner/model"
        assert files_metadata is True
        self.info_calls += 1
        siblings = [
            SimpleNamespace(
                rfilename=path,
                lfs=SimpleNamespace(sha256=sha256),
            )
            for path, sha256 in self.remote_hashes.items()
        ]
        return SimpleNamespace(sha=f"revision-{self.info_calls}", siblings=siblings)

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        component = kwargs["path_in_repo"]
        folder = kwargs["folder_path"]
        for pattern in kwargs["allow_patterns"]:
            path = folder / pattern
            if path.suffix == ".safetensors":
                self.remote_hashes[f"{component}/{pattern}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()


def test_verify_and_mark_batches_pending_weight_shards(tmp_path) -> None:
    component = "language_model"
    component_dir = tmp_path / component
    component_dir.mkdir()
    shards = []
    for index, content in enumerate((b"first", b"second"), start=1):
        name = f"model-{index:05d}.safetensors"
        (component_dir / name).write_bytes(content)
        shards.append(
            {
                "bytes": len(content),
                "file": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "tensors": [f"tensor.{index}"],
            }
        )
    manifest = {
        "component": component,
        "format": "mlx-minimax-music3-v1",
        "processed_source_files": ["source.safetensors"],
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
        "shards": shards,
        "source_files": ["source.safetensors"],
        "status": "complete",
        "total_size": sum(shard["bytes"] for shard in shards),
        "weight_map": {
            f"tensor.{index}": shard["file"]
            for index, shard in enumerate(shards, start=1)
        },
    }
    (component_dir / "conversion_manifest.json").write_text(json.dumps(manifest))
    (component_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": manifest["weight_map"]})
    )
    api = FakeHubApi()

    verified = verify_and_mark(api, "owner/model", tmp_path, component)

    assert api.info_calls == 2
    assert len(api.uploads) == 2
    assert api.uploads[0]["allow_patterns"] == [
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]
    assert api.uploads[1]["allow_patterns"] == [
        "conversion_manifest.json",
        "model.safetensors.index.json",
    ]
    assert all(shard["remote_verified"] for shard in verified["shards"])
    assert {shard["remote_revision"] for shard in verified["shards"]} == {"revision-2"}

