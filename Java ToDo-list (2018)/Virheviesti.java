package fi.todo.demo;

import java.util.Date;

public class Virheviesti {
    private String viesti;
    private Date aikaleima;

    public Virheviesti() {
        this("Tarkemmin määrittelemätön virhetilanne");
    }

    public Virheviesti(String viesti) {
        this.viesti = viesti;
        aikaleima = new Date();
    }

    public String getViesti() {
        return viesti;
    }

    public void setViesti(String viesti) {
        this.viesti = viesti;
    }

    public Date getAikaleima() {
        return aikaleima;
    }

    public void setAikaleima(Date aikaleima) {
        this.aikaleima = aikaleima;
    }
}
