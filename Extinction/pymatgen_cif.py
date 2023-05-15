"""
This file is copy/pasted from pymatgen cif.py for preprocessing cif files
"""
import re
import textwrap
import warnings
from collections import OrderedDict, deque
from monty.io import zopen
from monty.string import remove_non_ascii

class CifBlock:
    """
    Object for storing cif data. All data is stored in a single dictionary.
    Data inside loops are stored in lists in the data dictionary, and
    information on which keys are grouped together are stored in the loops
    attribute.
    """
    maxlen = 70  # not quite 80 so we can deal with semicolons and things

    def __init__(self, data, loops, header):
        """
        Args:
            data: dict or OrderedDict of data to go into the cif. Values should
                    be convertible to string, or lists of these if the key is
                    in a loop
            loops: list of lists of keys, grouped by which loop they should
                    appear in
            header: name of the block (appears after the data_ on the first
                line)
        """
        self.loops = loops
        self.data = data
        # AJ says: CIF Block names cannot be more than 75 characters or you
        # get an Exception
        self.header = header[:74]

    def __eq__(self, other):
        return self.loops == other.loops and self.data == other.data and self.header == other.header

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        """
        Returns the cif string for the data block
        """
        s = [f"data_{self.header}"]
        keys = self.data.keys()
        written = []
        for k in keys:
            if k in written:
                continue
            for l in self.loops:
                # search for a corresponding loop
                if k in l:
                    s.append(self._loop_to_string(l))
                    written.extend(l)
                    break
            if k not in written:
                # k didn't belong to a loop
                v = self._format_field(self.data[k])
                if len(k) + len(v) + 3 < self.maxlen:
                    s.append(f"{k}   {v}")
                else:
                    s.extend([k, v])
        return "\n".join(s)

    def _loop_to_string(self, loop):
        s = "loop_"
        for l in loop:
            s += "\n " + l
        for fields in zip(*[self.data[k] for k in loop]):
            line = "\n"
            for val in map(self._format_field, fields):
                if val[0] == ";":
                    s += line + "\n" + val
                    line = "\n"
                elif len(line) + len(val) + 2 < self.maxlen:
                    line += "  " + val
                else:
                    s += line
                    line = "\n  " + val
            s += line
        return s

    def _format_field(self, v):
        v = v.__str__().strip()
        if len(v) > self.maxlen:
            return ";\n" + textwrap.fill(v, self.maxlen) + "\n;"
        # add quotes if necessary
        if v == "":
            return '""'
        if (" " in v or v[0] == "_") and not (v[0] == "'" and v[-1] == "'") and not (v[0] == '"' and v[-1] == '"'):
            if "'" in v:
                q = '"'
            else:
                q = "'"
            v = q + v + q
        return v

    @classmethod
    def _process_string(cls, string):
        # remove comments
        string = re.sub(r"(\s|^)#.*$", "", string, flags=re.MULTILINE)
        # remove empty lines
        string = re.sub(r"^\s*\n", "", string, flags=re.MULTILINE)
        # remove non_ascii
        string = remove_non_ascii(string)
        # since line breaks in .cif files are mostly meaningless,
        # break up into a stream of tokens to parse, rejoining multiline
        # strings (between semicolons)
        q = deque()
        multiline = False
        ml = []
        # this regex splits on spaces, except when in quotes.
        # starting quotes must not be preceded by non-whitespace
        # (these get eaten by the first expression)
        # ending quotes must not be followed by non-whitespace
        p = re.compile(r"""([^'"\s][\S]*)|'(.*?)'(?!\S)|"(.*?)"(?!\S)""")
        for l in string.splitlines():
            if multiline:
                if l.startswith(";"):
                    multiline = False
                    q.append(("", "", "", " ".join(ml)))
                    ml = []
                    l = l[1:].strip()
                else:
                    ml.append(l)
                    continue
            if l.startswith(";"):
                multiline = True
                ml.append(l[1:].strip())
            else:
                for s in p.findall(l):
                    # s is tuple. location of the data in the tuple
                    # depends on whether it was quoted in the input
                    q.append(s)
        return q

    @classmethod
    def from_string(cls, string):
        """
        Reads CifBlock from string.

        :param string: String representation.
        :return: CifBlock
        """
        q = cls._process_string(string)
        header = q.popleft()[0][5:]
        data = OrderedDict()
        loops = []
        while q:
            s = q.popleft()
            # cif keys aren't in quotes, so show up in s[0]
            if s[0] == "_eof":
                break
            if s[0].startswith("_"):
                try:
                    data[s[0]] = "".join(q.popleft())
                except IndexError:
                    data[s[0]] = ""
            elif s[0].startswith("loop_"):
                columns = []
                items = []
                while q:
                    s = q[0]
                    if s[0].startswith("loop_") or not s[0].startswith("_"):
                        break
                    columns.append("".join(q.popleft()))
                    data[columns[-1]] = []
                while q:
                    s = q[0]
                    if s[0].startswith("loop_") or s[0].startswith("_"):
                        break
                    items.append("".join(q.popleft()))
                n = len(items) // len(columns)
                assert len(items) % n == 0
                loops.append(columns)
                for k, v in zip(columns * n, items):
                    data[k].append(v.strip())
            elif "".join(s).strip() != "":
                warnings.warn("Possible issue in cif file at line: {}".format("".join(s).strip()))
        return cls(data, loops, header)


class CifFile:
    """
    Reads and parses CifBlocks from a .cif file or string
    """

    def __init__(self, data, orig_string=None, comment=None):
        """
        Args:
            data (OrderedDict): Of CifBlock objects.å
            orig_string (str): The original cif string.
            comment (str): Comment string.
        """
        self.data = data
        self.orig_string = orig_string
        self.comment = comment or "# generated using pymatgen"

    def __str__(self):
        s = ["%s" % v for v in self.data.values()]
        return self.comment + "\n" + "\n".join(s) + "\n"

    @classmethod
    def from_string(cls, string):
        """
        Reads CifFile from a string.

        :param string: String representation.
        :return: CifFile
        """
        d = OrderedDict()
        for x in re.split(r"^\s*data_", "x\n" + string, flags=re.MULTILINE | re.DOTALL)[1:]:

            # Skip over Cif block that contains powder diffraction data.
            # Some elements in this block were missing from CIF files in
            # Springer materials/Pauling file DBs.
            # This block anyway does not contain any structure information, and
            # CifParser was also not parsing it.
            if "powder_pattern" in re.split(r"\n", x, 1)[0]:
                continue
            c = CifBlock.from_string("data_" + x)
            d[c.header] = c
        return cls(d, string)

    @classmethod
    def from_file(cls, filename):
        """
        Reads CifFile from a filename.

        :param filename: Filename
        :return: CifFile
        """
        with zopen(str(filename), "rt", errors="replace") as f:
            return cls.from_string(f.read())

