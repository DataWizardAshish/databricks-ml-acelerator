"""
Unity Catalog metadata reader.
Uses Databricks SDK (no Spark) — reads table/column metadata only, never raw data.
Auth falls through to ~/.databrickscfg DEFAULT profile when host/token are empty.
"""

from dataclasses import dataclass, field
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import TableType

from config.settings import get_settings


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
        """Compact column list for LLM context."""
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
    Never reads actual row data — only metadata APIs.
    """

    def __init__(self) -> None:
        settings = get_settings()
        # SDK picks up auth from env vars or ~/.databrickscfg DEFAULT profile
        kwargs: dict = {}
        if settings.databricks_host:
            kwargs["host"] = settings.databricks_host
        if settings.databricks_token and settings.auth_type == "pat":
            kwargs["token"] = settings.databricks_token

        self._client = WorkspaceClient(**kwargs)
        self._settings = settings

    def list_tables(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> list[TableInfo]:
        """
        List all tables in the given catalog.schema.
        Defaults to settings UC_CATALOG / UC_DISCOVERY_SCHEMA.
        """
        catalog = catalog or self._settings.uc_catalog
        schema = schema or self._settings.uc_discovery_schema

        tables: list[TableInfo] = []
        try:
            for t in self._client.tables.list(catalog_name=catalog, schema_name=schema):
                if t.table_type in (TableType.MANAGED, TableType.EXTERNAL):
                    table_info = TableInfo(
                        catalog=catalog,
                        schema=schema,
                        name=t.name or "",
                        full_name=t.full_name or f"{catalog}.{schema}.{t.name}",
                        table_type=t.table_type.value if t.table_type else "UNKNOWN",
                        comment=t.comment,
                        row_count=t.properties.get("delta.numLongRowsInserted")
                        if t.properties
                        else None,
                        columns=self._extract_columns(t),
                    )
                    tables.append(table_info)
        except Exception as e:
            raise RuntimeError(
                f"Failed to list tables in {catalog}.{schema}: {e}"
            ) from e

        return tables

    def get_table(self, full_name: str) -> TableInfo:
        """Fetch a single table by full_name (catalog.schema.table)."""
        try:
            t = self._client.tables.get(full_name=full_name)
        except Exception as e:
            raise RuntimeError(f"Failed to get table {full_name}: {e}") from e

        parts = full_name.split(".")
        return TableInfo(
            catalog=parts[0],
            schema=parts[1],
            name=parts[2],
            full_name=full_name,
            table_type=t.table_type.value if t.table_type else "UNKNOWN",
            comment=t.comment,
            columns=self._extract_columns(t),
        )

    def list_schemas(self, catalog: Optional[str] = None) -> list[str]:
        """List all schema names in a catalog."""
        catalog = catalog or self._settings.uc_catalog
        schemas = []
        for s in self._client.schemas.list(catalog_name=catalog):
            schemas.append(s.name or "")
        return [s for s in schemas if s]

    def _extract_columns(self, table) -> list[ColumnInfo]:
        columns = []
        if not table.columns:
            return columns
        for col in table.columns:
            columns.append(
                ColumnInfo(
                    name=col.name or "",
                    type_text=col.type_text or col.type_name.value
                    if col.type_name
                    else "UNKNOWN",
                    comment=col.comment,
                    nullable=col.nullable if col.nullable is not None else True,
                )
            )
        return columns

    def build_estate_summary(self, tables: list[TableInfo]) -> str:
        """
        Build a concise text summary of the data estate for the LLM.
        """
        lines = [
            f"Unity Catalog: {self._settings.uc_catalog}.{self._settings.uc_discovery_schema}",
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
