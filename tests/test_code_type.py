"""Tests for the production/test/generated/external/configuration code classifier."""

from __future__ import annotations

from argus.analysis.code_type import CodeType, classify_file


def test_classify_file_test_directory() -> None:
    assert classify_file("tests/test_orders.py") is CodeType.TEST
    assert classify_file("app/tests/helpers.py") is CodeType.TEST


def test_classify_file_test_naming_conventions() -> None:
    assert classify_file("test_orders.py") is CodeType.TEST
    assert classify_file("orders_test.py") is CodeType.TEST
    assert classify_file("conftest.py") is CodeType.TEST


def test_classify_file_production_default() -> None:
    assert classify_file("app/orders/service.py") is CodeType.PRODUCTION


def test_classify_file_generated() -> None:
    assert classify_file("app/migrations/0001_initial.py") is CodeType.GENERATED
    assert classify_file("proto/orders_pb2.py") is CodeType.GENERATED
    assert classify_file("proto/orders_pb2_grpc.py") is CodeType.GENERATED


def test_classify_file_configuration() -> None:
    assert classify_file("app/settings.py") is CodeType.CONFIGURATION
    assert classify_file("app/settings/production.py") is CodeType.CONFIGURATION


def test_classify_file_empty_is_unknown() -> None:
    assert classify_file("") is CodeType.UNKNOWN
