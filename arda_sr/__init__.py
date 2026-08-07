__all__ = ["ARDASRPipeline"]


def __getattr__(name):
    if name == "ARDASRPipeline":
        from .pipeline import ARDASRPipeline

        return ARDASRPipeline
    raise AttributeError(name)
