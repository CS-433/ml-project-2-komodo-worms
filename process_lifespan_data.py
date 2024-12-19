import os
import pandas as pd
from pathlib import Path
import argparse

# Final Processing Pipeline with Speed Column Removed
def process_dataset_final(file_path, chunk_size=10799, session_length=900, gap_seconds=5.5 * 3600):
    """
    Full pipeline to process the dataset:
    1. Split into chunks.
    2. Duplicate the first frame in each chunk.
    3. Split chunks into 900-frame sessions.
    4. Combine sessions into one dataset.
    5. Add timestamps.
    6. Remove the 'Speed' column if present.
    
    Parameters:
        file_path (str): Path to the input CSV file.
        chunk_size (int): Number of frames in each chunk before restarting.
        session_length (int): Number of frames per session.
        gap_seconds (float): Gap in seconds between sessions.
    
    Returns:
        pd.DataFrame: Fully processed dataset.
    """
    # Step 1: Load the data
    data = pd.read_csv(file_path)

    # Step 2: Split into chunks
    def split_into_chunks(data, chunk_size):
        return [data.iloc[i:i + chunk_size].copy() for i in range(0, len(data), chunk_size)]
    
    chunks = split_into_chunks(data, chunk_size)

    # Step 3: Duplicate the last frame in each chunk
    def duplicate_last_frame(chunks):
        for i, chunk in enumerate(chunks):
            last_frame = chunk.iloc[-1].copy()
            chunks[i] = pd.concat([chunk, pd.DataFrame([last_frame])], ignore_index=True)
        return chunks

    chunks_with_duplicated_frames = duplicate_last_frame(chunks)

    # Step 4: Split chunks into sessions
    def split_chunks_into_sessions(chunks, session_length):
        sessions = []
        for chunk in chunks:
            for i in range(0, len(chunk), session_length):
                session = chunk.iloc[i:i + session_length].copy()
                sessions.append(session)
        return sessions

    sessions = split_chunks_into_sessions(chunks_with_duplicated_frames, session_length)

    # Step 5: Combine sessions into one dataset
    def combine_sessions_with_continuous_frames(sessions):
        combined = pd.concat(sessions, ignore_index=True)
        combined['Frame'] = combined.index  # Make frame numbering continuous
        return combined

    combined_data = combine_sessions_with_continuous_frames(sessions)

    # Step 6: Add timestamps
    def add_timestamps(data, session_length, gap_seconds):
        timestamps = []
        for i in range(len(data)):
            session_index = i // session_length
            frame_within_session = i % session_length
            timestamp = session_index * (session_length * 2 + gap_seconds) + frame_within_session * 2
            timestamps.append(timestamp)
        data['Timestamp'] = timestamps
        return data

    combined_data = add_timestamps(combined_data, session_length, gap_seconds)

    # Step 7: Remove the 'Speed' column if present
    if 'Speed' in combined_data.columns:
        combined_data.drop(columns=['Speed'], inplace=True)

    return combined_data

def process_all_files(input_dir, output_dir, chunk_size=10799, session_length=900, gap_seconds=5.5 * 3600):
    """
    Process all files in the input directory and save to output directory.
    
    Parameters:
        input_dir (Path): Base directory containing all treatment groups
        output_dir (Path): Directory to save processed files
        chunk_size (int): Number of frames in each chunk before restarting
        session_length (int): Number of frames per session
        gap_seconds (float): Gap in seconds between sessions
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing parameters:")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Session length: {session_length}")
    print(f"  Gap between sessions: {gap_seconds/3600:.1f} hours")
    
    # Process each treatment group directory
    for treatment_dir in input_dir.iterdir():
        if treatment_dir.is_dir():
            treatment_name = treatment_dir.name
            print(f"\nProcessing treatment group: {treatment_name}")
            
            # Create treatment-specific output directory
            treatment_output_dir = output_dir / treatment_name
            treatment_output_dir.mkdir(exist_ok=True)
            
            # Process each CSV file in the treatment directory
            for csv_file in treatment_dir.glob('*.csv'):
                print(f"Processing file: {csv_file.name}")
                try:
                    # Process the file
                    processed_data = process_dataset_final(
                        str(csv_file),
                        chunk_size=chunk_size,
                        session_length=session_length,
                        gap_seconds=gap_seconds
                    )
                    
                    # Create output filename
                    output_filename = f"processed_{csv_file.name}"
                    output_path = treatment_output_dir / output_filename
                    
                    # Save processed data
                    processed_data.to_csv(output_path, index=False)
                    print(f"Successfully processed and saved: {output_filename}")
                    
                except Exception as e:
                    print(f"Error processing {csv_file.name}: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process lifespan data by splitting into sessions.')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing raw data')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--chunk-size', type=int, default=10799, help='Number of frames in each chunk')
    parser.add_argument('--session-length', type=int, default=900, help='Number of frames per session')
    parser.add_argument('--gap-hours', type=float, default=5.5, help='Gap between sessions in hours')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Convert gap hours to seconds
    gap_seconds = args.gap_hours * 3600
    
    # Process all files
    process_all_files(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        session_length=args.session_length,
        gap_seconds=gap_seconds
    )

if __name__ == "__main__":
    main() 