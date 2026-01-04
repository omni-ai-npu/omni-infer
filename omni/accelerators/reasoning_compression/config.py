class ThinkCompressDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"No such think compress variable: {name}") from e

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(f"No such think compress variable: {name}") from e


ThinkCompressDict = ThinkCompressDict()
ThinkCompressDict.reasoner_early_think_stopping_enabled = False