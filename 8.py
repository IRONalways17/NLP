def simulate_fa(input_string):
    """
    Simulates a finite automaton that accepts strings containing
    the substring '110' repeated (e.g., 110110110)
    """
    # Define states
    # q0: Initial state, haven't seen any matching characters
    # q1: Seen a '1'
    # q2: Seen '11'
    # q3: Seen '110', accepting state and restart pattern
    
    current_state = 'q0'
    
    print(f"Input: {input_string}")
    print("Simulation:")
    
    # Process each character
    for i, char in enumerate(input_string):
        print(f"  State: {current_state}, Reading: {char}", end=" -> ")
        
        if current_state == 'q0':
            if char == '1':
                current_state = 'q1'
            # else stay in q0
        
        elif current_state == 'q1':
            if char == '1':
                current_state = 'q2'
            else:
                current_state = 'q0'
        
        elif current_state == 'q2':
            if char == '0':
                current_state = 'q3'
            else:
                current_state = 'q2'  # Another '1' keeps us in q2
        
        elif current_state == 'q3':
            if char == '1':
                current_state = 'q1'
            else:
                current_state = 'q0'
        
        print(f"New state: {current_state}")
    
    # Check if the pattern 110 was found (i.e., reached q3 at any point)
    accepts = input_string.find('110') != -1
    print(f"\nFinal state: {current_state}")
    print(f"String {'accepted' if accepts else 'rejected'}")
    return accepts

# Example usage
test_string = "110110110"
simulate_fa(test_string)