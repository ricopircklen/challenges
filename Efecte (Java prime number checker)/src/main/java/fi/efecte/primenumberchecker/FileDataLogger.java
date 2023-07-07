package fi.efecte.primenumberchecker;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class FileDataLogger implements DataLogger {
    private FileWriter fw;
    private BufferedWriter bw;
    private PrintWriter pw;

    public FileDataLogger(String fileName) throws IOException {
        fw = new FileWriter(fileName, true); //fileWriter
        bw = new BufferedWriter(fw); //bufferedWriter
        pw = new PrintWriter(bw); //printWriter
    }

    public void write(String data) {
        pw.println(data);
        pw.flush();
    }

   public void close() throws IOException {
        pw.close();
        bw.close();
        fw.close();
    }
}