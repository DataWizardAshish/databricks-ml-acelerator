"""
Unity Catalog metadata reader.
Uses SQL connector (Path A — OBO token) to query information_schema.
UC row-level policies and column masking are enforced as the end user.
No Spark, no compute, no raw data reads.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

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
    Reads Unity Catalog metadata via information_schema SQL queries.
    Path A: sql.connect with OBO token — queries run as the end user.
    """

    def __init__(self, ctx: WorkspaceContext) -> None:
        self._ctx = ctx

    def list_tables(self, catalog: Optional[str] = None, schema: Optional[str] = None) -> list[TableInfo]:
        catalog = catalog or self._ctx.catalog
        schema = schema or self._ctx.schema

        conn = self._ctx.get_sql_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT table_name, table_type, comment
                    FROM `{catalog}`.information_schema.tables
                    WHERE table_schema = '{schema}'
                      AND table_type IN ('MANAGED', 'EXTERNAL')
                    ORDER BY table_name
                """)
                table_rows = cur.fetchall()

                cur.execute(f"""
                    SELECT table_name, column_name, data_type, comment, is_nullable
                    FROM `{catalog}`.information_schema.columns
                    WHERE table_schema = '{schema}'
                    ORDER BY table_name, ordinal_position
                """)
                col_rows = cur.fetchall()
        except Exception as e:
            raise RuntimeError(f"Failed to list tables in {catalog}.{schema}: {e}") from e
        finally:
            conn.close()

        cols_by_table: dict[str, list[ColumnInfo]] = defaultdict(list)
        for row in col_rows:
            cols_by_table[row[0]].append(ColumnInfo(
                name=row[1],
                type_text=row[2],
                comment=row[3],
                nullable=(row[4] == "YES") if row[4] is not None else True,
            ))

        tables: list[TableInfo] = []
        for row in table_rows:
            name = row[0]
            tables.append(TableInfo(
                catalog=catalog,
                schema=schema,
                name=name,
                full_name=f"{catalog}.{schema}.{name}",
                table_type=row[1] or "UNKNOWN",
                comment=row[2],
                columns=cols_by_table.get(name, []),
            ))
        return tables

    def get_table(self, full_name: str) -> TableInfo:
        parts = full_name.split(".")
        catalog, schema, name = parts[0], parts[1], parts[2]

        conn = self._ctx.get_sql_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT table_type, comment
                    FROM `{catalog}`.information_schema.tables
                    WHERE table_schema = '{schema}' AND table_name = '{name}'
                """)
                t_row = cur.fetchone()

                cur.execute(f"""
                    SELECT column_name, data_type, comment, is_nullable
                    FROM `{catalog}`.information_schema.columns
                    WHERE table_schema = '{schema}' AND table_name = '{name}'
                    ORDER BY ordinal_position
                """)
                col_rows = cur.fetchall()
        except Exception as e:
            raise RuntimeError(f"Failed to get table {full_name}: {e}") from e
        finally:
            conn.close()

        columns = [
            ColumnInfo(
                name=r[0], type_text=r[1], comment=r[2],
                nullable=(r[3] == "YES") if r[3] is not None else True,
            )
            for r in col_rows
        ]
        return TableInfo(
            catalog=catalog, schema=schema, name=name, full_name=full_name,
            table_type=t_row[0] if t_row else "UNKNOWN",
            comment=t_row[1] if t_row else None,
            columns=columns,
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
