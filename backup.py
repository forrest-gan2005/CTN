"""Snapshots of the team's SQLite databases while the team runs, and restoring one when it is stopped."""
import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DATA = Path(__file__).parent / "data"
BACKUPS = DATA / "backups"
KEEP = 12  # snapshots kept; at one every 15 minutes, the last three hours
NAMES = ("bots.db", "npc.db", "accounts.db", "usage.db")
MARKER = "RESTORED"  # left in the data directory by a restore, for the team's next start


def db_file(conn: sqlite3.Connection) -> str:
    """The file a connection has open ("" for an in-memory database)."""
    return conn.execute("PRAGMA database_list").fetchone()[2]


def snapshot(files: dict, root: Path = BACKUPS, keep: int = KEEP) -> Path:
    """Copy each database (name -> path) with SQLite's online backup, which is consistent even while another
    connection writes (WAL included), check each copy's integrity, and keep the newest `keep` snapshots.
    A snapshot directory only appears once complete. Each database is copied on its own, so they are
    not one atomic team-wide moment; the team's startup reconciliation covers the difference."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    tmp = root / f".{stamp}"
    tmp.mkdir(parents=True)
    for name, path in files.items():
        src, dst = sqlite3.connect(path), sqlite3.connect(tmp / name)
        try:
            src.backup(dst)
            if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError(f"snapshot of {name} failed its integrity check")
        finally:
            src.close()
            dst.close()
    done = root / stamp
    tmp.rename(done)
    for old in snapshots(root)[:-keep]:
        shutil.rmtree(old)
    return done


def snapshots(root: Path = BACKUPS) -> list:
    """Complete snapshots, oldest first."""
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) if root.exists() else []


def restore(snap: Path, data: Path = DATA) -> list:
    """Put a snapshot's databases back in place. Only with the team stopped: the old WAL and shared-memory files
    go too, or SQLite would replay them over the restored file."""
    restored = []
    for src in sorted(snap.glob("*.db")):
        for suffix in ("", "-wal", "-shm"):
            (data / (src.name + suffix)).unlink(missing_ok=True)
        shutil.copy2(src, data / src.name)
        restored.append(src.name)
    (data / MARKER).write_text(snap.name)
    return restored


def latest(root: Path = BACKUPS) -> Optional[Path]:
    found = snapshots(root)
    return found[-1] if found else None


def _demo() -> None:
    import tempfile
    import threading

    with tempfile.TemporaryDirectory() as tmp:
        data, root = Path(tmp) / "data", Path(tmp) / "backups"
        data.mkdir()
        live = sqlite3.connect(data / "bots.db", check_same_thread=False)
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("CREATE TABLE t (n INTEGER)")
        live.commit()
        stop = threading.Event()

        def writer() -> None:
            n = 0
            while not stop.is_set():
                with live:
                    live.execute("INSERT INTO t VALUES (?)", (n,))
                n += 1

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            snaps = [snapshot({"bots.db": str(data / "bots.db")}, root, keep=2) for _ in range(3)]
        finally:
            stop.set()
            thread.join()
        assert snapshots(root) == snaps[1:], "older snapshots pruned"
        copied = sqlite3.connect(snaps[-1] / "bots.db").execute("SELECT COUNT(*), MAX(n) FROM t").fetchone()
        assert copied[0] == copied[1] + 1, "a consistent prefix of the writes"
        total = live.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        live.close()
        assert restore(snaps[-1], data) == ["bots.db"] and not (data / "bots.db-wal").exists()
        assert (data / MARKER).read_text() == snaps[-1].name
        assert sqlite3.connect(data / "bots.db").execute("SELECT COUNT(*) FROM t").fetchone()[0] == copied[0] <= total
    print("backup ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("list", help="the snapshots kept")
    rs = sub.add_parser("restore", help="put a snapshot back (stop the team first)")
    rs.add_argument("name", nargs="?", help="snapshot directory name; the newest if left out")
    args = parser.parse_args()
    if args.cmd == "list":
        for snap in snapshots():
            print(snap.name, " ".join(sorted(p.name for p in snap.glob("*.db"))))
    elif args.cmd == "restore":
        snap = BACKUPS / args.name if args.name else latest()
        if snap is None or not snap.is_dir():
            sys.exit("no such snapshot")
        print(f"restoring {snap.name}: {', '.join(restore(snap))}; start the team again to reconcile")
    else:
        _demo()
