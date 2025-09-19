package dntb2;

import org.junit.Test;
import static org.junit.Assert.*;

public class HeapBasedTest extends HeapBased {

    @Test
    public void testEmptyList() {
        System.out.println("Running testEmptyList...");
        ListCell result = sort(null);
        printList(result);
        assertNull("Failed: Sorting an empty list should return null.", result);
    }

    @Test
    public void testSingleElement() {
        System.out.println("Running testSingleElement...");
        ListCell node = new ListCell();
        node.val = 42;
        node.nxt = null;

        ListCell result = sort(node);
        printList(result);
        assertNotNull("Failed: Result should not be null.", result);
        assertEquals("Failed: The only element should be 42.", 42, result.val);
        assertNull("Failed: There should be no further nodes.", result.nxt);
    }

    @Test
    public void testSortedList() {
        System.out.println("Running testSortedList...");
        // Already sorted list: 1 -> 2 -> 3 -> 4 -> 5
        int[] values = {1, 2, 3, 4, 5};
        ListCell head = createList(values);
        ListCell result = sort(head);
        printList(result);
        assertTrue("Failed: The list should be sorted in non-decreasing order.", isSorted(result));
    }

    @Test
    public void testPartiallySortedList() {
        System.out.println("Running testPartiallySortedList...");
        // Two sorted segments: 1, 3, 5 and 2, 4, 6 (the original list is not globally sorted)
        int[] values = {1, 3, 5, 2, 4, 6};
        ListCell head = createList(values);
        ListCell result = sort(head);
        printList(result);
        assertTrue("Failed: The list should be sorted in non-decreasing order.", isSorted(result));
    }

    @Test
    public void testPartiallySortedList2() {
        System.out.println("Running testPartiallySortedList2...");
        int[] values = {1, 3, 5, 2, 4, 6, 7, 8, 9, 10, 4, 6, 10, 52, 21, 23, 45, -1, 0, 1, 2};
        ListCell head = createList(values);
        ListCell result = sort(head);
        printList(result);
        assertTrue("Failed: The list should be sorted in non-decreasing order.", isSorted(result));
    }

    @Test
    public void testPartiallySortedList3() {
        System.out.println("Running testPartiallySortedList3...");
        int[] values = {10, 14, 124, 2151, 21152, 1, 2, 3, 4, 5, 210, 12, 32, 412, 421, 54, 45, 61, 21};
        ListCell head = createList(values);
        ListCell result = sort(head);
        printList(result);
        assertTrue("Failed: The list should be sorted in non-decreasing order.", isSorted(result));
    }

    @Test
    public void testReverseSortedList() {
        System.out.println("Running testReverseSortedList...");
        // Reverse order list: 5 -> 4 -> 3 -> 2 -> 1
        int[] values = {5, 4, 3, 2, 1};
        ListCell head = createList(values);
        ListCell result = sort(head);
        printList(result);
        assertTrue("Failed: The list should be sorted in non-decreasing order.", isSorted(result));
    }

    /**
     * Helper method to create a linked list from an array of integers.
     */
    private ListCell createList(int[] values) {
        if (values == null || values.length == 0)
            return null;
        ListCell head = new ListCell();
        head.val = values[0];
        ListCell current = head;
        for (int i = 1; i < values.length; i++) {
            ListCell node = new ListCell();
            node.val = values[i];
            current.nxt = node;
            current = node;
        }
        return head;
    }

    /**
     * Helper method to verify that the linked list is sorted in non-decreasing order.
     */
    private boolean isSorted(ListCell head) {
        if (head == null)
            return true;
        int prev = head.val;
        head = head.nxt;
        while (head != null) {
            if (head.val < prev) {
                return false;
            }
            prev = head.val;
            head = head.nxt;
        }
        return true;
    }
}
