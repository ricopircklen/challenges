package fi.todo.demo.dao;

import fi.todo.demo.Todo;

import java.util.List;
import java.util.Optional;

public interface TodoDao {
    List<Todo> haeKaikki();
    int lisaa(Todo todo);
    Todo poista(int id);
}
