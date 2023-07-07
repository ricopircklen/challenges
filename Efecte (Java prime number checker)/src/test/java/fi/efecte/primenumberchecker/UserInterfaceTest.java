package fi.efecte.primenumberchecker;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class UserInterfaceTest {

    @Test
    public void validateInput_MinusOne_False() {
        UserInterface ui = new UserInterface();
        boolean isAnalyzableInput = ui.validateInput("-1");
        assertFalse(isAnalyzableInput);
    }

    @Test
    public void validateInput_Two_True() {
        UserInterface ui = new UserInterface();
        boolean isAnalyzableInput = ui.validateInput("2");
        assertTrue(isAnalyzableInput);
    }

    @Test
    public void validateInput_Text_False() {
        UserInterface ui = new UserInterface();
        boolean isAnalyzableInput = ui.validateInput("Text");
        assertFalse(isAnalyzableInput);
    }

    @Test
    public void validateInput_Empty_False() {
        UserInterface ui = new UserInterface();
        boolean isAnalyzableInput = ui.validateInput("");
        assertFalse(isAnalyzableInput);
    }
}
