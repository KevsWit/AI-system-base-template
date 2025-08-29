from typing import Literal

Domain = Literal["GYM", "CLIENTS", "UNKNOWN"]

class Router:
    def route(self, relevance_label: str) -> Domain:
        if relevance_label == "GYM":
            return "GYM"
        if relevance_label == "CLIENTS":
            return "CLIENTS"
        return "UNKNOWN"
