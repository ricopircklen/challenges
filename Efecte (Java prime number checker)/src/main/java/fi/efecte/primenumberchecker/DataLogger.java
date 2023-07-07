package fi.efecte.primenumberchecker;

import java.io.IOException;

public interface DataLogger {
    void write(String data);
    void close() throws IOException;
}
