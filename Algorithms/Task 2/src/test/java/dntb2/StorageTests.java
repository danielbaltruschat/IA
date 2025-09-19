package dntb2;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class StorageTests {

    @Test
    public void testInitialiseTooSmall() {
        Storage storage = new Storage();
        // Expect an OutOfMemoryError when the memory length is too small (5 < 6).
        assertThrows(OutOfMemoryError.class, () -> storage.initialise(5));
    }

    @Test
    public void testInitialiseNormal() {
        Storage storage = new Storage();
        // Should not throw any exception.
        assertDoesNotThrow(() -> storage.initialise(20));
    }

    @Test
    public void testMallocSuccess() {
        Storage storage = new Storage();
        storage.initialise(30);
        int ptr = storage.malloc(4);  // Request allocation for 4 integers.
        assertNotEquals(-1, ptr, "malloc should return a valid pointer");
    }

    @Test
    public void testMallocExhaustion() {
        Storage storage = new Storage();
        storage.initialise(30);
        int ptr;
        int count = 0;
        while ((ptr = storage.malloc(1)) != -1) {
            count++;
        }
        // After exhaustion, malloc should return -1 and at least one block should have been allocated.
        assertTrue(count > 0, "At least one block should be allocated before exhaustion");
        assertEquals(-1, storage.malloc(1), "After exhaustion, malloc should return -1");
    }

    @Test
    public void testFreeAndReallocate() {
        Storage storage = new Storage();
        storage.initialise(40);
        int ptr1 = storage.malloc(4);
        int ptr2 = storage.malloc(3);
        int ptr3 = storage.malloc(2);

        // Free the middle block.
        storage.free(ptr2);

        // Try allocating another block of the same size as ptr2.
        int ptr4 = storage.malloc(3);
        assertNotEquals(-1, ptr4, "Re-allocation into freed space should succeed");
    }

    @Test
    public void testDoubleFree() {
        Storage storage = new Storage();
        storage.initialise(30);
        int ptr = storage.malloc(3);
        storage.free(ptr);
        // Free it again.
        storage.free(ptr);
        int ptr2 = storage.malloc(3);
        assertNotEquals(-1, ptr2, "Allocation after double free should succeed");
    }

    @Test
    public void testMergeFreeBlocks() {
        Storage storage = new Storage();
        storage.initialise(50);
        int ptr1 = storage.malloc(5);
        int ptr2 = storage.malloc(5);
        int ptr3 = storage.malloc(5);

        // Free two adjacent blocks.
        storage.free(ptr1);
        storage.free(ptr2);

        // Attempt to allocate a block that requires the merged free space.
        int ptrLarge = storage.malloc(8);
        assertNotEquals(-1, ptrLarge, "Allocation in merged free block should succeed");
    }

    @Test
    public void testLargeAllocation() {
        Storage storage = new Storage();
        storage.initialise(100);
        // With length 100, available free space should be 95 ints.
        int ptr = storage.malloc(95);
        assertNotEquals(-1, ptr, "Large allocation should succeed");
    }

    @Test
    public void testAllocAfterComplexFreeMerging() {
        Storage storage = new Storage();
        storage.initialise(60);
        int ptr1 = storage.malloc(5);
        int ptr2 = storage.malloc(5);
        int ptr3 = storage.malloc(5);

        // Free the middle and last block to force merging.
        storage.free(ptr2);
        storage.free(ptr3);

        // Now try to allocate a block that requires combined space.
        int ptrLarge = storage.malloc(8);
        assertNotEquals(-1, ptrLarge, "Allocation after complex free merging should succeed");
    }

    @Test
    public void testSequentialAllocsAndFrees() {
        Storage storage = new Storage();
        storage.initialise(100);
        int ptr1 = storage.malloc(10);
        int ptr2 = storage.malloc(10);
        int ptr3 = storage.malloc(10);

        // Free the first allocated block and allocate a smaller block.
        storage.free(ptr1);
        int ptr4 = storage.malloc(5);

        // Free the second block and allocate a block that may require splitting.
        storage.free(ptr2);
        int ptr5 = storage.malloc(8);

        // Free the third block and try another allocation.
        storage.free(ptr3);
        int ptr6 = storage.malloc(12);

        assertNotEquals(-1, ptr4, "Allocation for ptr4 should succeed");
        assertNotEquals(-1, ptr5, "Allocation for ptr5 should succeed");
        assertNotEquals(-1, ptr6, "Allocation for ptr6 should succeed");
    }

    @Test
    public void testMinimumMemoryAllocation() {
        Storage storage = new Storage();
        // With length 6, the available free block size is exactly 1.
        assertDoesNotThrow(() -> storage.initialise(6), "Initialise with minimum memory should not throw");
        int ptr = storage.malloc(1);
        assertNotEquals(-1, ptr, "Malloc on minimum memory should succeed");
    }

    @Test
    public void testCompleteFreeAndReallocate() {
        Storage storage = new Storage();
        storage.initialise(50);
        int ptr1 = storage.malloc(5);
        int ptr2 = storage.malloc(5);
        int ptr3 = storage.malloc(5);

        // Free all allocated blocks.
        storage.free(ptr1);
        storage.free(ptr2);
        storage.free(ptr3);

        // Now allocate a block that should fit into the merged free space.
        int ptrLarge = storage.malloc(12);
        assertNotEquals(-1, ptrLarge, "Allocation after freeing all blocks should succeed");
    }

    @Test
    public void testRepeatedAllocFree() {
        Storage storage = new Storage();
        storage.initialise(100);
        int iterations = 50;
        for (int i = 0; i < iterations; i++) {
            int ptr = storage.malloc(5);
            assertNotEquals(-1, ptr, "Iteration " + i + ": malloc should succeed");
            storage.free(ptr);
        }
    }

    @Test
    public void testEdgeCaseLargeAndSmallAllocation() {
        Storage storage = new Storage();
        storage.initialise(100);
        int ptrLarge = storage.malloc(80);  // Allocate a large block.
        assertNotEquals(-1, ptrLarge, "Large allocation should succeed");
        storage.free(ptrLarge);
        int count = 0;
        int ptr;
        while ((ptr = storage.malloc(5)) != -1) {
            count++;
        }
        assertTrue(count > 0, "After freeing large block, small allocations should succeed");
    }

    @Test
    public void testInterleavedAllocFree() {
        Storage storage = new Storage();
        storage.initialise(120);
        int ptrA = storage.malloc(10);
        int ptrB = storage.malloc(10);
        int ptrC = storage.malloc(10);
        int ptrD = storage.malloc(10);

        // Free second block.
        storage.free(ptrB);
        int ptrE = storage.malloc(8);

        // Free first and third blocks.
        storage.free(ptrA);
        storage.free(ptrC);
        int ptrF = storage.malloc(15);

        // Free fourth block.
        storage.free(ptrD);
        int ptrG = storage.malloc(12);

        assertNotEquals(-1, ptrE, "Allocation for ptrE should succeed");
        assertNotEquals(-1, ptrF, "Allocation for ptrF should succeed");
        assertNotEquals(-1, ptrG, "Allocation for ptrG should succeed");
    }

    @Test
    public void testAllocationBoundary() {
        int length = 30;
        Storage storage = new Storage();
        storage.initialise(length);
        // The available free block size is length - 5.
        int blockSize = length - 5;
        int ptr = storage.malloc(blockSize);
        assertNotEquals(-1, ptr, "Allocation at boundary condition should succeed");
    }
}
