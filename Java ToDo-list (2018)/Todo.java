package fi.todo.demo;

public class Todo {
    private int id;
    private String otsikko;
    private String teksti;

    public Todo() {
    }

    public Todo(String otsikko) {
        this.otsikko = otsikko;
    }

    public Todo(String otsikko, String teksti) {
        this.otsikko = otsikko;
        this.teksti = teksti;
    }

    public String getOtsikko() {
        return otsikko;
    }

    public void setOtsikko(String otsikko) {
        this.otsikko = otsikko;
    }

    public String getTeksti() {
        return teksti;
    }

    public void setTeksti(String teksti) {
        this.teksti = teksti;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    @Override
    public String toString() {
        return "Todo{" +
                "otsikko='" + otsikko + '\'' +
                ", teksti='" + teksti + '\'' +
                '}';
    }
}