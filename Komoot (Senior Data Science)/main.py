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

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    try:
        newsletter_generator = NewsletterGenerator(input_file, output_file)
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
