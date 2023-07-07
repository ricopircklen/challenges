package fi.todo.demo.dao.jdbc;

import fi.todo.demo.Todo;
import fi.todo.demo.dao.TodoDao;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.sql.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

@Service
@Qualifier("jdbc")
public class TodoDaoJdbcImpl implements TodoDao {
    private Connection con;

    public TodoDaoJdbcImpl() throws SQLException, ClassNotFoundException {
        Class.forName("org.postgresql.Driver");
        System.out.println("Ajuri ladattu");

        String url = "jdbc:postgresql://localhost:5432/todot";
        con = DriverManager.getConnection(url,
                "postgres", "postgres");
        System.out.println("Yhteys saatu");
    }

    @Override
    public List<Todo> haeKaikki() {
        String sql = "SELECT * FROM todot";
        List<Todo> haetut = new ArrayList<>();
        try (PreparedStatement pstmt = con.prepareStatement(sql)) {
            for (ResultSet rs = pstmt.executeQuery(); rs.next(); ) {
                Todo a = new Todo();
                a.setId(rs.getInt("id"));
                a.setOtsikko(rs.getString("otsikko"));
                a.setTeksti(rs.getString("teksti"));
                haetut.add(a);
            }
        } catch (SQLException e) {
            e.printStackTrace();
            return Collections.EMPTY_LIST; // tai null kun nyt virhetilanteesta on kyse...
        }
        return haetut;
    }

    @Override
    public int lisaa(Todo todo) {
        int avain = -1;
        String sql = "INSERT INTO todot(otsikko, teksti) VALUES (?,?)";
        try (PreparedStatement pr = con.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS)) {
            pr.setString(1, todo.getOtsikko());
            pr.setString(2, todo.getTeksti());
            pr.execute();
            ResultSet avaimet = pr.getGeneratedKeys();
            while (avaimet.next()) {
                avain = avaimet.getInt(1);
                todo.setId(avain);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return avain;
    }

    @Override
    public Todo poista(int id) {
        Todo poistettu = new Todo();
        String sel = "SELECT * FROM todot WHERE id = ?";
        try (PreparedStatement prs = con.prepareStatement(sel)) {
            prs.setInt(1, id);
            prs.execute();
            ResultSet rs = prs.executeQuery();
            while (rs.next()) {
                poistettu.setId(rs.getInt("id"));
                poistettu.setOtsikko(rs.getString("otsikko"));
                poistettu.setTeksti(rs.getString("teksti"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        String sql = "DELETE FROM todot WHERE id = ?";
        try (PreparedStatement pr = con.prepareStatement(sql)) {
            pr.setInt(1, id);
            pr.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return poistettu;
    }
}

