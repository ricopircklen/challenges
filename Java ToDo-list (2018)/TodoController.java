package fi.todo.demo;

import fi.todo.demo.dao.TodoDao;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.net.URI;
import java.util.List;

@RestController
@RequestMapping("/api/todot")
@CrossOrigin
public class TodoController {
    private TodoDao dao;

    @Autowired
    public TodoController(@Qualifier("jdbc") TodoDao dao) {
        this.dao = dao;
    }
    @GetMapping("")
    public List<Todo> listaaKaikki() {
        List<Todo> kaikki = dao.haeKaikki();
        System.out.printf("Haetaan listaa, alkioita: %d kpl\n", kaikki.size());
        System.out.println(kaikki);
        return kaikki;
    }

    @PostMapping("")
    public ResponseEntity<?> luoUuusi(@RequestBody Todo uus) {
        System.out.println("****** Luodaan uutta: " + uus);
        int id = dao.lisaa(uus);
        System.out.println("****** Luotu uusi: " + uus);
        URI location = ServletUriComponentsBuilder
                .fromCurrentRequest()
                .path("/{id}")
                .buildAndExpand(id)
                .toUri();
        return ResponseEntity.created(location).body(uus);
    }
    @DeleteMapping("/{id}")
    public ResponseEntity<?> poista(@PathVariable int id) {
        Todo poistettu = dao.poista(id);
        if (poistettu != null)
            return ResponseEntity.ok(poistettu);
        return ResponseEntity
                .status(HttpStatus.NOT_FOUND)
                .body(new Virheviesti(String.format("Id %d ei ole olemassa: ei poistettu", id)));
    }
}

