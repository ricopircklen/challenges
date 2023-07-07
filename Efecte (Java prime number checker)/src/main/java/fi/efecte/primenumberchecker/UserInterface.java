package fi.efecte.primenumberchecker;

import java.io.IOException;
import java.util.Scanner;

import static fi.efecte.primenumberchecker.PrimeNumberChecker.checkIfPrimeOrNot;
import static fi.efecte.primenumberchecker.PrimeNumberChecker.primeResultToString;

class UserInterface {
    private Scanner scanner;
    private DataLogger dataLogger;

    UserInterface() {
        this(new ConsoleDataLogger());
    }

    UserInterface(DataLogger dataLogger) {
        this.scanner = new Scanner(System.in);
        this.dataLogger = dataLogger;
    }

    void run() {
        System.out.println("This is a prime number checker. Press 'q' to quit.");
        while (true) {
            System.out.println("Input number");
            String input = scanner.nextLine();
            if ("q".equals(input)) {
                closeSystems(scanner, dataLogger);
                break;
            }
            if (validateInput(input)) {
                int inputValue = Integer.parseInt(input);
                int analyzedValue = checkIfPrimeOrNot(inputValue);
                dataLogger.write(primeResultToString(inputValue, analyzedValue));
            }
        }
    }

    private void closeSystems(Scanner scanner, DataLogger dataLogger) {
        scanner.close();
        try {
            dataLogger.close();
        } catch (IOException e) {
            System.out.println("Something went wrong while closing the file logging system.");
        }
    }

    /**
     * Method to check that input is an integer with value >1.
     *
     * @return true for integers >1, false for non-integers and integers <=1.
     */
    boolean validateInput(String input) {
        String message ='"' + input + '"' + " is not a valid input. Please, type an integer between 2-2147483647.";
        try {
            int inputValue = Integer.parseInt(input);
            if (inputValue <= 1) {
               // throw new IllegalArgumentException();
                System.err.println(message);
                return false;
            }
            return true;
        } catch (IllegalArgumentException e) {
            System.out.println(message);
        }
        return false;
    }


}
