package fi.efecte.primenumberchecker;

public class ConsoleDataLogger implements DataLogger {
    public void write(String data) {
        System.out.println(data);
    }

    public void close() {
    }
}

