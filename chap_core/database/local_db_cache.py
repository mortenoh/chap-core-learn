# Improvement Suggestions:
# 1. Comprehensive Docstrings: Add module, class, and method docstrings explaining purpose, parameters, and assumptions.
# 2. Refine `__getitem__` Behavior: Ensure `__getitem__` raises KeyError or returns Optional[SQLModel] and is documented.
# 3. Generalize `value` Type: Consider generalizing the `value: float` in `__setitem__` if cache is for diverse models.
# 4. Decorator Flexibility: Improve `decorate` flexibility regarding decorated function signature and cache key formation.
# 5. Error Handling in Decorator: Add error handling in the decorator's wrapper for function calls and cache operations.

from typing import Any, Callable, Optional, Tuple, Type

from sqlmodel import Session, SQLModel, select


class LocalDbCache:
    """
    A caching layer that uses a SQLModel table for persistence.

    This class provides a dictionary-like interface (`__contains__`, `__getitem__`,
    `__setitem__`) to cache key-value pairs, where keys are tuples of
    (period_id, region_id) and values are stored in a specified SQLModel table.
    It assumes the provided SQLModel `model` has `period_id`, `region_id`,
    and `value` attributes.
    """

    def __init__(self, session: Session, model: Type[SQLModel]):
        """
        Initializes the LocalDbCache.

        Args:
            session: The SQLModel session to use for database interactions.
            model: The SQLModel class definition to use for storing cache entries.
                   This model must have `period_id`, `region_id`, and `value` fields.
        """
        self._session: Session = session
        self._model: Type[SQLModel] = model

    def __contains__(self, item: Tuple[str, str]) -> bool:
        """
        Checks if an item exists in the cache.

        Args:
            item: A tuple of (period_id, region_id).

        Returns:
            True if the item exists in the cache, False otherwise.
        """
        period_id, region_id = item
        # Assuming self._model has period_id and region_id attributes
        statement = select(self._model).where(
            self._model.period_id == period_id,
            self._model.region_id == region_id,  # type: ignore[attr-defined]
        )
        result = self._session.exec(statement).first()
        return bool(result)

    def __getitem__(self, item: Tuple[str, str]) -> Optional[SQLModel]:
        """
        Retrieves an item from the cache.

        Args:
            item: A tuple of (period_id, region_id).

        Returns:
            The cached SQLModel instance if found, otherwise None.
            Note: Standard dicts raise KeyError; this returns None for simplicity here.
        """
        period_id, region_id = item
        # Assuming self._model has period_id and region_id attributes
        statement = select(self._model).where(
            self._model.period_id == period_id,
            self._model.region_id == region_id,  # type: ignore[attr-defined]
        )
        result = self._session.exec(statement).first()
        return result

    def __setitem__(self, key: Tuple[str, str], value: float) -> None:
        """
        Adds or updates an item in the cache.

        Args:
            key: A tuple of (period_id, region_id).
            value: The float value to cache. Assumes the model's 'value' field accepts float.
        """
        period_id, region_id = key
        # Assumes model can be instantiated with period_id, region_id, value
        new_entry = self._model(period_id=period_id, region_id=region_id, value=value)  # type: ignore[call-arg]
        self._session.add(new_entry)
        self._session.commit()

    @classmethod
    def decorate(cls, model: Type[SQLModel]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Class method decorator factory to apply caching to a function.

        The decorated function is expected to have `period_id` and `region_id`
        as its first two positional arguments, and accept a `session` keyword
        argument for caching to be active.

        Args:
            model: The SQLModel class to use for caching results.

        Returns:
            A decorator that wraps a function with caching logic.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """The actual decorator that wraps the function."""

            def wrapper(period_id: str, region_id: str, *args: Any, **kwargs: Any) -> Any:
                """
                Wrapper function that implements the caching logic.

                If 'session' is in kwargs, it attempts to retrieve from cache.
                If not found, it calls the original function and caches its result.
                """
                if "session" not in kwargs:
                    return func(period_id, region_id, *args, **kwargs)

                session: Session = kwargs.pop("session")
                cache_instance = cls(session, model)

                cache_key = (period_id, region_id)
                if cache_key in cache_instance:
                    cached_model_instance = cache_instance[cache_key]
                    if cached_model_instance:
                        # Assuming the relevant value is stored in a 'value' attribute
                        return getattr(cached_model_instance, "value", None)  # type: ignore[attr-defined]
                    # Fall through to calling the function if None (though __contains__ should prevent this)

                # Call the original function
                value_to_cache = func(period_id, region_id, *args, **kwargs)

                # Cache the result
                # This assumes value_to_cache is compatible with what __setitem__ expects (float)
                # and that the model's 'value' field can store it.
                if isinstance(value_to_cache, float):  # Added type check for safety
                    cache_instance[cache_key] = value_to_cache
                else:
                    # Handle or log cases where the returned value isn't a float,
                    # as __setitem__ currently expects a float.
                    # For now, we'll just return the value without caching if not float.
                    pass  # Or log a warning: logger.warning("Value not float, not caching.")

                return value_to_cache

            return wrapper

        return decorator
