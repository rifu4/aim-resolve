"""Configuration management for KL optimization runs."""

import os
from copy import deepcopy

import numpy as np

from ..fast_resolve.opt_kl import fast_optimize_kl
from .opt_kl import optimize_kl
from .samples import domain_keys
from .util import (
    add_dicts,
    clean_dict,
    clean_reps,
    eval_list,
    eval_string,
    extend_reps,
    fun2mode,
    get_it,
    is_or_contains_type,
    merge_dicts,
    split_its,
)
from .yml import yaml_load, yaml_save


class OptimizeKLConfig:
    """Class to initialize a nifty optimization from a single or multiple yaml configuration files."""

    def __init__(self, sections, builders):
        """
        Initialize the OptimizeKLConfig class.

        Parameters
        ----------
        sections : dict
            Configuration sections.
        builders : dict
            Dictionary of builder functions.
        """
        self.sections = dict(sections)
        self.interpret_base()
        self.interpret_link()
        self.interpret_mode()
        self.interpret_reps()
        self.join_opt_stages()
        self.builders = (
            builders(self.sections) if callable(builders) else dict(builders)
        )

    @classmethod
    def from_file(cls, fname, builders):
        """
        Import a config file and instantiate the class.

        Parameters
        ----------
        fname : str
            File name of the config file that is imported.
        builders : dict
            Dictionary of functions that are used to instantiate e.g. operators.
        """
        sections = yaml_load(fname)

        return cls(sections, builders)

    def to_file(self, fname):
        """
        Write configuration in standardized form to file.

        Parameters
        ----------
        fname : str
            Path to which the config shall be written.
        """
        dct = clean_dict(self.sections)
        dct["opt.0"] = clean_reps(dct["opt.0"], simplify=True)
        dct_lst = split_its(dct)

        yaml_save(dct_lst, fname)

        return

    def optimize_kl(self, **kwargs):
        """
        Do the inference and save the config file to the output directory.

        Parameters
        ----------
        kwargs : dict
            Additional parameters for the `optimize_kl` function (e.g. callback).
        """
        dct = dict(self)

        os.makedirs(dct["odir"], exist_ok=True)
        self.to_file(os.path.join(dct["odir"], "opt.yml"))

        if self.mode == "fast":
            return fast_optimize_kl(**dct, **kwargs)
        else:
            return optimize_kl(**dct, **kwargs)

    def interpret_base(self):
        """Replace the `base` entries in all (sub)sections by the content of the section it points to."""
        dct = self.sections

        for sec in dct:
            dct[sec] = get_base(dct[sec], dct)

        return

    def interpret_link(self):
        """Replace the `->` entries in all (sub)sections by the content (string) of the section key it points to."""
        dct = self.sections

        for sec in dct:
            dct[sec] = get_link(dct[sec], dct)

        return

    def interpret_mode(self):
        """Check and get the mode of the likelihood and set the optimization parameters accordingly."""
        dct = self.sections

        dct = fun2mode(dct)

        self.opt_params = dict(
            n_iter=["n_total_iterations"],
            static=["odir", "position_or_samples", "key", "resume"],
            needed_dyn=[
                "likelihood",
                "n_samples",
                "draw_linear_kwargs",
                "nonlinearly_update_kwargs",
                "kl_kwargs",
                "sample_mode",
            ],
            option_dyn=["constants", "point_estimates", "transitions"],
        )

        modes = set()
        for opt_key in filter(lambda x: x[:4] == "opt.", dct.keys()):
            lh_key = dct[opt_key].get("likelihood", None)
            if lh_key is None:
                raise KeyError(
                    f"key `likelihood` is missing in opt-section `{opt_key}`"
                )
            if isinstance(lh_key, list):
                lh_key = lh_key[0]
            if lh_key[1:] not in dct:
                raise KeyError(f"section `{lh_key[1:]}` is missing in `self.sections`")
            mode = dct[lh_key[1:]].get("mode")
            if mode is None:
                raise KeyError(
                    f"key `mode` is missing in likelihood section `{lh_key[1:]}`"
                )
            modes.add(mode)

        if len(modes) != 1:
            raise RuntimeError(
                f"All likelihood modes have to be the same, but got modes `{modes}`."
            )

        self.mode = modes.pop()
        if self.mode == "fast":
            self.opt_params["n_iter"] = ["n_major_iterations"]
            self.opt_params["needed_dyn"] += ["n_minor_iterations"]

        return

    def interpret_reps(self):
        """Expand the repetitions of all sections starting with `opt.`. Check if all necessary keys are present."""
        dct = self.sections

        for opt_key in filter(lambda x: x[:4] == "opt.", dct.keys()):
            for key in self.opt_params["n_iter"] + self.opt_params["needed_dyn"]:
                if key not in dct[opt_key]:
                    raise KeyError(f"key `{key}` is missing in opt section `{opt_key}`")
            for key in self.opt_params["option_dyn"]:
                if key not in dct[opt_key]:
                    dct[opt_key][key] = None

            if self.mode == "fast":
                [
                    dct[opt_key].pop(key)
                    for key in ["n_total_iterations"]
                    if key in dct[opt_key]
                ]
                n_major = dct[opt_key]["n_major_iterations"]
                minor_key = "n_minor_iterations"
                n_minor = get_reps({minor_key: dct[opt_key][minor_key]}, n_major)[
                    minor_key
                ]
                dct[opt_key] = get_reps(dct[opt_key], n_major, n_minor)
            else:
                [
                    dct[opt_key].pop(key)
                    for key in ["n_major_iterations", "n_minor_iterations"]
                    if key in dct[opt_key]
                ]
                dct[opt_key] = get_reps(
                    dct[opt_key], dct[opt_key]["n_total_iterations"]
                )

        return

    def join_opt_stages(self):
        """
        Join the repetitions for all sections starting with `opt.` to a single section called `opt.0`.

        Sort the sections in ascending order, add their leaves and clean up the `opt.` section.
        Remove the old `opt.` sections.
        """
        dct = self.sections

        opt_keys = sorted(
            (k for k in dct.keys() if k.startswith("opt.")),
            key=lambda k: int(k.split(".")[1]),
        )
        dct["opt.0"] = add_dicts(*[dct[k] for k in opt_keys])
        dct["opt.0"] = clean_reps(dct["opt.0"], simplify=False)

        for k in filter(lambda k: k != "opt.0", opt_keys):
            del dct[k]

        return

    def make_callable(self, sec, key=None):
        """
        Turn the section repetition lists into callable functions of the iteration number.
        Instantiate all references indicated by `=` using the builders dictionary.
        """

        def fun(it):
            val = get_it(sec, it)
            if key in ["constants", "point_estimates"]:
                val = self.get_constants_or_point_estimates(val, it)
            elif isinstance(val, str):
                if len(val) > 1 and val.startswith("="):  # is reference
                    val = self.instantiate_sec(val[1:])
            return val

        if is_or_contains_type(sec, list):
            return fun
        else:
            return fun(0)

    def instantiate_sec(self, sec):
        """
        Instantiate an object that is described by a section in the config file by looking up
        the section key in the `self._builders` dictionary and call the respective function.
        """
        dct = deepcopy(self.sections[sec])

        # Instantiate all references (also in subsections)
        for key, val in dct.items():
            if isinstance(val, str):
                if len(val) > 1 and val[0] == "=":  # is reference
                    dct[key] = self.instantiate_sec(val[1:])

        # Plug into builders dictionary
        if sec in self.builders:
            return self.builders[sec](**dct)
        raise RuntimeError(f"Provide build routine for `{sec}` in builders dictionary")

    def get_constants_or_point_estimates(self, cpe, it):
        """
        Get both the constants and point estimates for the current iteration. Given a model section name,
        it adds all parameter keys of that model component. For a `~` in front of the name, it includes
        all likelihood parameter keys except the ones of the model component.
        """
        match cpe:
            case (
                None
                | [
                    None,
                ]
                | (None,)
                | []
                | ()
            ):
                return None
            case str():
                cpe = [cpe]
            case tuple():
                cpe = list(cpe)

        match (all("~" in c for c in cpe), any("~" in c for c in cpe)):
            case (True, _):
                neg = True
            case (False, False):
                neg = False
            case (False, True):
                raise ValueError(
                    "Negation `~` has to be used for all or none of the constants/point_estimates"
                )

        if self.mode == "fast":
            minor_cs = np.cumsum(self.sections["opt.0"]["n_minor_iterations"])
            it = np.searchsorted(minor_cs, it, side="right")

        lh_sec = get_it(self.sections["opt.0"]["likelihood"], it)
        m_keys = domain_keys(self.instantiate_sec(lh_sec[1:])["model"])

        cpe_new = set()
        for c in cpe:
            match c.replace("=", "").strip("~"):
                case s if s in m_keys:
                    cpe_new.add(s)
                case s if s in self.sections:
                    c_keys = domain_keys(self.instantiate_sec(s))
                    cpe_new.update(k for k in c_keys if k in m_keys)
                case _:
                    raise ValueError(f"Cannot find `{c}` in sections or `{m_keys}`.")

        if neg:
            return tuple(m_keys - cpe_new)
        return tuple(cpe_new)

    def __iter__(self):
        """Enable conversion to `dict` to pass everyting to the `optimize_kl` function."""
        # static
        sopt = self.sections["opt"]
        for key in self.opt_params["static"]:
            if key in sopt:
                yield key, sopt[key]

        # dynamic
        sdyn = self.sections["opt.0"]
        for key in self.opt_params["n_iter"]:
            if key in sdyn:
                yield key, sdyn[key]
        for key in self.opt_params["needed_dyn"] + self.opt_params["option_dyn"]:
            if key in sdyn:
                yield key, self.make_callable(sdyn[key], key)

    def __str__(self):
        """Return a human-readable string representation of all configuration sections."""
        s = []
        for key, val in self.sections.items():
            s += [key]
            s += [f"  {kk}: {vv}" for kk, vv in val.items()]
            s += [""]
        return "\n".join(s)

    def __eq__(self, other):
        """Check equality based on sections and builders."""
        for a in "sections", "builders":
            if getattr(self, a) != getattr(other, a):
                return False
        return True


def get_base(sec_dct, dct, key_lst=[]):
    """Recursively replace the `base` entries in all (sub)sections by the content of the section it points to."""
    for key, val in sec_dct.items():
        if len(key_lst) != len(set(key_lst)):
            raise RuntimeError("You are trying a base-loop. Please do not do that :(")

        if isinstance(val, dict):
            sec_dct[key] = merge_dicts(
                [sec_dct[key], get_base(val, dct, key_lst + [key])]
            )

        elif key.startswith("base"):
            sub_dct = dct.copy()
            sub_keys = ["self.sections"]
            while "/" in val:
                sub, val = val.split("/", 2)
                if sub not in sub_dct:
                    err_sub = (
                        f"section `{sub_keys[-1]}`"
                        if len(sub_keys) > 1
                        else f"`{sub_keys[-1]}`"
                    )
                    raise RuntimeError(
                        f"the referred section `{sub}` does not exist in {err_sub}"
                    )
                sub_dct = sub_dct[sub]
                sub_keys += [sub]
            if val not in sub_dct:
                err_sub = (
                    f"section `{sub_keys[-1]}`"
                    if len(sub_keys) > 1
                    else f"`{sub_keys[-1]}`"
                )
                raise RuntimeError(
                    f"the referred section `{val}` does not exist in {err_sub}"
                )
            sec_dct = merge_dicts(
                [get_base(sub_dct[val], dct, key_lst + sub_keys[1:] + [val]), sec_dct]
            )

    return sec_dct


def get_link(sec_dct, dct):
    """Recursively replace the `->` entries in all (sub)sections by the content (string) of the section key it points to."""
    for key, val in sec_dct.items():
        if isinstance(val, dict):
            sec_dct[key] = merge_dicts([sec_dct[key], get_link(val, dct)])

        elif isinstance(val, str) and "->" in val:
            val = map(lambda x: x.strip(), val.split("+"))
            new_val = ""
            for v in val:
                if v.startswith("->"):
                    v = v[2:].strip()
                    sub_dct = deepcopy(dct)
                    sub_keys = ["self.sections"]
                    while "/" in v:
                        sub, v = v.split("/", 2)
                        if sub not in sub_dct:
                            err_sub = (
                                f"section `{sub_keys[-1]}`"
                                if len(sub_keys) > 1
                                else f"`{sub_keys[-1]}`"
                            )
                            raise RuntimeError(
                                f"the referred section `{sub}` does not exist in {err_sub}"
                            )
                        sub_dct = sub_dct[sub]
                    if v not in sub_dct:
                        err_sub = (
                            f"section `{sub_keys[-1]}`"
                            if len(sub_keys) > 1
                            else f"`{sub_keys[-1]}`"
                        )
                        raise RuntimeError(
                            f"the referred section `{v}` does not exist in {err_sub}"
                        )
                    v = sub_dct[v]
                    if not isinstance(v, str):
                        raise ValueError(
                            f"the referred section value `{v}` has to be a string."
                        )
                    elif "->" in v:
                        raise ValueError("recursive links not allowed for now")
                new_val = os.path.join(new_val, v.strip("/"))
            sec_dct[key] = new_val

    return sec_dct


def get_reps(sec_dct, total_it, minor_it=None):
    """Recursively expand the repetitions of all sections starting with `opt.`."""
    for key, val in sec_dct.items():
        if isinstance(val, dict):
            sec_dct[key] = get_reps(val, total_it, minor_it)
            continue

        elif key in ["n_total_iterations", "n_major_iterations"]:
            if not isinstance(val, int) or val < 1:
                raise TypeError(f"`{key}` has to be of type `int` and larger than 0")
            sec_dct[key] = val
            continue

        elif isinstance(val, str):
            val = eval_string(val)

        if not isinstance(val, list) or val == []:
            val = [val]

        if isinstance(val, list):
            val = eval_list(val)
            if key in ["constants", "point_estimates", "transitions"]:
                val = extend_reps(val, total_it, None)
            else:
                val = extend_reps(val, total_it)

            if minor_it and key not in [
                "n_minor_iterations",
                "likelihood",
                "transitions",
            ]:
                for i, mi in enumerate(minor_it):
                    vi = val[i]
                    if not isinstance(vi, list) or vi == []:
                        vi = [vi]
                    val[i] = extend_reps(vi, mi)

            sec_dct[key] = val

    return sec_dct
