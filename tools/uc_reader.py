"""
Unity Catalog metadata reader.
Uses Databricks SDK REST APIs — no Spark, no compute, no raw data reads.
Accepts a WorkspaceContext so each request can target a different workspace.
"""

from dataclasses import dataclass, field
from typing import Optional

from databricks.sdk.service.catalog import TableType

from tools.workspace_context import WorkspaceContext


@dataclass
class ColumnInfo:
    name: str
    type_text: str
    comment: Optional[str] = None
    nullable: bool = True


@dataclass
class TableInfo:
    catalog: str
    schema: str
    name: str
    full_name: str
    table_type: str
    comment: Optional[str] = None
    row_count: Optional[int] = None
    columns: list[ColumnInfo] = field(default_factory=list)

    @property
    def column_summary(self) -> str:
        parts = []
        for col in self.columns:
            desc = f"{col.name} ({col.type_text})"
            if col.comment:
                desc += f" — {col.comment}"
            parts.append(desc)
        return ", ".join(parts)


class UCReader:
    """
    Reads Unity Catalog metadata for a given catalog + schema.
    Never reads actual row data — metadata APIs only.
    """

    def __init__(self, ctx: WorkspaceContext) -> None:
        self._ctx = ctx
        self._client = ctx.get_workspace_client()

    def list_tables(self, catalog: Optional[str] = None, schema: Optional[str] = None) -> list[TableInfo]:
        catalog = catalog or self._ctx.catalog
        schema = schema or self._ctx.schema

        tables: list[TableInfo] = []
        try:
            for t in self._client.tables.list(catalog_name=catalog, schema_name=schema):
                if t.table_type in (TableType.MANAGED, TableType.EXTERNAL):
                    tables.append(TableInfo(
                        catalog=catalog,
                        schema=schema,
                        name=t.name or "",
                        full_name=t.full_name or f"{catalog}.{schema}.{t.name}",
                        table_type=t.table_type.value if t.table_type else "UNKNOWN",
                        comment=t.comment,
                        row_count=t.properties.get("delta.numLongRowsInserted") if t.properties else None,
                        columns=self._extract_columns(t),
                    ))
        except Exception as e:
            raise RuntimeError(f"Failed to list tables in {catalog}.{schema}: {e}") from e

        return tables

    def get_table(self, full_name: str) -> TableInfo:
        try:
            t = self._client.tables.get(full_name=full_name)
        except Exception as e:
            raise RuntimeError(f"Failed to get table {full_name}: {e}") from e
        parts = full_name.split(".")
        return TableInfo(
            catalog=parts[0], schema=parts[1], name=parts[2], full_name=full_name,
            table_type=t.table_type.value if t.table_type else "UNKNOWN",
            comment=t.comment, columns=self._extract_columns(t),
        )

    def build_estate_summary(self, tables: list[TableInfo]) -> str:
        lines = [
            f"Unity Catalog: {self._ctx.catalog}.{self._ctx.schema}",
            f"Total tables discovered: {len(tables)}",
            "",
        ]
        for t in tables:
            lines.append(f"TABLE: {t.full_name} [{t.table_type}]")
            if t.comment:
                lines.append(f"  Description: {t.comment}")
            if t.row_count:
                lines.append(f"  Row count (approx): {t.row_count}")
            if t.columns:
                lines.append(f"  Columns ({len(t.columns)}): {t.column_summary}")
            lines.append("")
        return "\n".join(lines)

    def _extract_columns(self, table) -> list[ColumnInfo]:
        columns = []
        if not table.columns:
            return columns
        for col in table.columns:
            columns.append(ColumnInfo(
                name=col.name or "",
                type_text=col.type_text or (col.type_name.value if col.type_name else "UNKNOWN"),
                comment=col.comment,
                nullable=col.nullable if col.nullable is not None else True,
            ))
        return columns
