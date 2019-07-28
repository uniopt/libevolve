# -*- coding: utf-8 -*-

import pytest
from libevolve.common._base import *


class TestInputParam:
    def test_constructor(self):
        a = InputParam(name="a")
        a = InputParam(name="a", param_value=2)
        assert a._val == 2, "param value not assigned properly"

    def test_current_value(self):
        a = InputParam(name="a", param_value=2)
        assert a.current_value == 2, "param value not assigned properly"


class TestEvoParam:
    def test_constructor(self):
        pass
