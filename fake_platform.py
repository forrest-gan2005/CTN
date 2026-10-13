import itertools
from datetime import datetime
from typing import Optional


class FakePlatform:
    def __init__(self) -> None:
        self.posts = {}          # post_id -> post dict
        self.following = {}      # username -> set of usernames they follow
        self.notifications = {}  # username -> list of notification dicts
        self._ids = itertools.count(1)

    def post(self, username: str, content: str, parent_id: Optional[int] = None) -> dict:
        if parent_id is not None:
            parent = self._get(parent_id)
        post = {
            "id": next(self._ids),
            "author": username,
            "content": content,
            "parent_id": parent_id,
            "likes": set(),
            "reposts": set(),
            "created_at": datetime.now().isoformat(),
        }
        self.posts[post["id"]] = post
        if parent_id is not None:
            self._notify(parent["author"], "reply", username, post["id"])
        return post

    def reply(self, username: str, parent_id: int, content: str) -> dict:
        return self.post(username, content, parent_id=parent_id)

    def like(self, username: str, post_id: int) -> bool:
        """Idempotent: returns False if already liked, never toggles."""
        post = self._get(post_id)
        if username in post["likes"]:
            return False
        post["likes"].add(username)
        self._notify(post["author"], "like", username, post_id)
        return True

    def unlike(self, username: str, post_id: int) -> bool:
        post = self._get(post_id)
        if username not in post["likes"]:
            return False
        post["likes"].remove(username)
        return True

    def repost(self, username: str, post_id: int) -> bool:
        post = self._get(post_id)
        if username in post["reposts"]:
            return False
        post["reposts"].add(username)
        self._notify(post["author"], "repost", username, post_id)
        return True

    def follow(self, username: str, target: str) -> bool:
        if username == target:
            return False
        followed = self.following.setdefault(username, set())
        if target in followed:
            return False
        followed.add(target)
        self._notify(target, "follow", username, None)
        return True

    def unfollow(self, username: str, target: str) -> bool:
        followed = self.following.get(username, set())
        if target not in followed:
            return False
        followed.remove(target)
        return True

    def get_post(self, post_id: int) -> dict:
        return self._get(post_id)

    def feed_latest(self, n: int = 10) -> list:
        return sorted(self.posts.values(), key=lambda p: p["id"], reverse=True)[:n]

    def feed_home(self, username: str, n: int = 10) -> list:
        followed = self.following.get(username, set())
        followed_posts = [p for p in self.posts.values() if p["author"] in followed]
        return sorted(followed_posts, key=lambda p: p["id"], reverse=True)[:n]

    def replies(self, post_id: int) -> list:
        self._get(post_id)
        return [p for p in self.posts.values() if p["parent_id"] == post_id]

    def search(self, query: str, n: int = 10) -> list:
        q = query.lower()
        hits = [p for p in self.posts.values() if q in p["content"].lower()]
        return sorted(hits, key=lambda p: p["id"], reverse=True)[:n]

    def get_notifications(self, username: str, clear: bool = True) -> list:
        """clear=True marks them as read."""
        notes = self.notifications.get(username, [])
        if clear:
            self.notifications[username] = []
        return notes

    def _get(self, post_id: int) -> dict:
        if post_id not in self.posts:
            raise KeyError(f"post {post_id} does not exist")
        return self.posts[post_id]

    def _notify(self, to: str, kind: str, actor: str, post_id: Optional[int]) -> None:
        if to == actor:
            return
        self.notifications.setdefault(to, []).append({
            "kind": kind,
            "actor": actor,
            "post_id": post_id,
            "created_at": datetime.now().isoformat(),
        })


if __name__ == "__main__":
    p = FakePlatform()
    a = p.post("alice", "Clinics on the outer islands are already stretched.")
    p.like("bob", a["id"])
    assert p.like("bob", a["id"]) is False
    p.reply("bob", a["id"], "My aunt waited 3 days for a nurse in 2021.")
    p.follow("carol", "alice")
    p.repost("carol", a["id"])

    assert len(a["likes"]) == 1
    assert [x["author"] for x in p.feed_home("carol")] == ["alice"]
    assert {n["kind"] for n in p.get_notifications("alice")} == {"like", "reply", "follow", "repost"}
    assert p.get_notifications("alice") == []
    assert len(p.search("NURSE")) == 1
    try:
        p.like("bob", 999)
    except KeyError:
        pass
    print("ok:", p.feed_latest())
