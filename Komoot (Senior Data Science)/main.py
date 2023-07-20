import argparse
import logging
import sys
from src.newsletter_generator import NewsletterGenerator
from src.exceptions import FileFormatError, MissingColumnsError, InsufficientRowsError


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate a newsletter CSV file for group cycling events.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('output_file', help='Path to the output CSV file')
    parser.add_argument('--max_group_size', type=int, default=40,
                        help='Maximum group size for clustering. Default is 40.')
    parser.add_argument('--max_chunk_size', type=int, default=10000,
                        help='Maximum chunk size for reading large input files. Default is 10000.')
    parser.add_argument('--min_cluster_size', type=int, default=5,
                        help='Minimum cluster size for OPTICS clustering. Default is 5.')
    parser.add_argument('--xi', type=float, default=0.01,
                        help='The parameter for the OPTICS algorithm, specifying a minimum steepness on the reachability plot. Default is 0.01.')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    max_group_size = args.max_group_size
    max_chunk_size = args.max_chunk_size
    min_cluster_size = args.min_cluster_size
    xi = args.xi

    try:
        newsletter_generator = NewsletterGenerator(
            input_file, output_file, max_group_size, max_chunk_size, min_cluster_size, xi)
        newsletter_generator.generate_newsletter()
    except (FileFormatError, MissingColumnsError, InsufficientRowsError) as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.exception(
            "An error occurred during newsletter CSV generation:", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
