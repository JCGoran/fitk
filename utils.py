class Parameter:
    """
    Class that contains everything to do with the parameter (its fiducial value, etc.)
    """
    def __init__(
        self,
        value : float = 0,
        # TODO other data
    ):
        self._fiducial = value

    @property
    def fiducial(self):
        return self._fiducial

    @fiducial.setter
    def fiducial(self, value):
        self._fiducial = value

    def __repr__(self):
        return f'fiducial: {self.fiducial}'


class CustomSet:
    """
    A custom set-like class which preserves ordering of elements.
    Internally, it's a Python `dict` with all values mapping to `None`.
    """

    def __init__(self, data):
        self._data = {key : Parameter() for key in data}

    @property
    def data(self):
        return tuple(self._data.keys())

    def index(self, idx : int):
        return list(self._data.keys()).index(idx)

    def __getitem__(self, key):
        if not isinstance(key, int) and key not in self._data.keys():
            raise KeyError(f'Key {key} not found')
        if isinstance(key, int):
            return list(self._data.keys())[key]
        return self._data[key]

    def __setitem__(self, key, value):
        if key not in self._data.keys():
            raise KeyError(f'Key {key} not found')
        if value != key and value in self._data.keys():
            raise ValueError(f'The value {value} already exists')
        if value != key:
            self._data = {_ if _ != key else value : __ for _, __ in self._data.items()}

    def __iter__(self):
        return self._data.__iter__()

    def __repr__(self):
        return str(tuple(self._data.keys()))

    def __len__(self):
        return len(self._data.keys())

    def __eq__(self, other):
        return tuple(self._data.keys()) == other

    def __hash__(self):
        return hash(self.data)
