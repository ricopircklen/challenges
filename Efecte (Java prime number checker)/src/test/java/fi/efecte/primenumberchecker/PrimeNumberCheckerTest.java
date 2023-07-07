package fi.efecte.primenumberchecker;

import org.junit.Test;

import static fi.efecte.primenumberchecker.PrimeNumberChecker.checkIfPrimeOrNot;
import static fi.efecte.primenumberchecker.PrimeNumberChecker.primeResultToString;
import static org.junit.Assert.assertEquals;

public class PrimeNumberCheckerTest {

    @Test
    public void isPrime_7_True() {
        int isPrime = checkIfPrimeOrNot(7);
        assertEquals("7 is prime.", 1, isPrime);
    }

    @Test
    public void isPrime_9_False() {
        int isPrime = checkIfPrimeOrNot(9);
        assertEquals("9 is not prime. It is divisible by 3", 3, isPrime);
    }

    @Test
    public void isPrime_3412151_True() {
        int isPrime = checkIfPrimeOrNot(3412151);
        assertEquals("3412151 is prime.", 1, isPrime);
    }

    @Test
    public void isPrime_36123516_False() {
        int isPrime = checkIfPrimeOrNot(36123516);
        assertEquals("36123516 is not prime. It is divisible by 2.", 2, isPrime);
    }

    @Test
    public void isPrime_0_False() {
        int isPrime = checkIfPrimeOrNot(0);
        assertEquals("0 is not prime.", -1, isPrime);
    }

    @Test
    public void isPrime_1_False() {
        int isPrime = checkIfPrimeOrNot(1);
        assertEquals("1 is not prime.", -1, isPrime);
    }

    @Test
    public void isPrime_minus7_False() {
        int isPrime = checkIfPrimeOrNot(-7);
        assertEquals("-7 is not prime.", -1, isPrime);
    }

    @Test
    public void isPrime_2147483647_True() {
        int isPrime = checkIfPrimeOrNot(2147483647);
        assertEquals("2147483647 is prime.", 1, isPrime);
    }

    @Test
    public void isPrime_minus2147483648_False() {
        int isPrime = checkIfPrimeOrNot(-2147483648);
        assertEquals("-2147483648 is not prime.", -1, isPrime);
    }

    @Test
    public void primeResultToString_AnalyzedValueMinusOne_IsNotPrime() {
        String primeResultAsString = primeResultToString(-10, -1);
        assertEquals("Analyzed value -1, input not prime.", "-10 is not prime.", primeResultAsString);
    }

    @Test
    public void primeResultToString_AnalyzedValueOne_IsPrime() {
        String primeResultAsString = primeResultToString(7, 1);
        assertEquals("Analyzed value 1, input is prime.", "7 is prime.", primeResultAsString);
    }

    @Test
    public void primeResultToString_AnalyzedValueThree_IsNotPrime() {
        String primeResultAsString = primeResultToString(9, 3);
        assertEquals("Analyzed value >1, input not prime and divisible by analyzed value.", "9 is not prime. It is divisible by 3.", primeResultAsString);
    }

}
