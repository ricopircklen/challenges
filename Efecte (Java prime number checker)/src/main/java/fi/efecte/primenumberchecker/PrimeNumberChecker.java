package fi.efecte.primenumberchecker;

public class PrimeNumberChecker {
    /**
     * Calculates whether an integer is a prime number or not.
     *
     * @return 1 if value is prime; -1 if value smaller or equal to 1 (i.e. not prime);
     * The smallest divider if value is not prime (e.g. 3 when analyzable value is 9).
     */
    static int checkIfPrimeOrNot(int value) {
        if (value <= 1) {
            return -1;
        }
        for (int i = 2; i <= Math.sqrt(value); i++) {
            if (value % i == 0) {
                return i;
            }
        }
        return 1;
    }

    /**
     * Method to translate checkIfPrimeOrNot-method's result to easily readable String.
     *
     * @param inputValue    value that was checked if prime or not
     * @param analyzedValue value returned from checkIfPrimeOrNot-method
     */
    static String primeResultToString(int inputValue, int analyzedValue) {
        if (analyzedValue == -1) {
            return inputValue + " is not prime.";
        } else if (analyzedValue == 1) {
            return inputValue + " is prime.";
        } else {
            return inputValue + " is not prime. It is divisible by " + analyzedValue + ".";
        }
    }
}

