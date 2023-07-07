package fi.efecte.primenumberchecker;

import java.io.IOException;

public class App {
    public static void main(String[] args) {
        if (args.length >= 1) {
            try {
                DataLogger fileDataLogger = new FileDataLogger(args[0]);
                UserInterface ui = new UserInterface(fileDataLogger);
                ui.run();
            } catch (IOException e) {
                System.out.println("Something went wrong. Please, check the file given as an argument.");
                System.out.println(e.getMessage());
            } finally {
                  //  fileDataLogger.close();
            }
        } else {
            UserInterface ui = new UserInterface();
            ui.run();
        }
    }
}
