import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.config import validate_typed_config
    from vipe.streams.base import StreamList

    typed_args = validate_typed_config(args)

    # Gather all video streams
    stream_list = StreamList.make(typed_args.streams)

    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # Build the pipeline once and reuse it across streams so that its cached
    # models (depth, GeoCalib, TrackAnything networks) are loaded a single time.
    pipeline = make_pipeline(typed_args.pipeline)

    # Process each video stream (or multiview rig)
    logger = configure_logging()
    for stream_idx in range(len(stream_list)):
        video_data = stream_list[stream_idx]
        if hasattr(video_data, "name"):
            name = video_data.name()  # type: ignore[union-attr]
        else:
            name = f"stream_{stream_idx}"
        logger.info(f"Processing {name} ({stream_idx + 1} / {len(stream_list)})")
        pipeline.run(video_data)
        logger.info(f"Finished processing {name}")


if __name__ == "__main__":
    run()
