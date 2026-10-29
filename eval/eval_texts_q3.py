"""Qwen3-TTS quality-gate texts: 24 English (same as the VoxCPM2 gate), 4 Spanish, 4 German. Numbers as words."""
EN = ['Hello, this is a quick check of the voice. Your order will be delivered by tomorrow evening, please keep your phone reachable.',
 'Good morning. This is a reminder that your payment of two thousand five hundred rupees is due on the fifth of October.',
 'Thank you for calling. Our team has registered your request and will contact you shortly with an update.',
 'I am calling from the bank regarding your credit card statement for the month of September.',
 'Could you please confirm your date of birth and the last four digits of your registered mobile number?',
 'The weather today is pleasant, with a light breeze and a few clouds drifting across the sky.',
 'We noticed an unusual transaction on your account and wanted to verify it with you before proceeding.',
 'Your appointment with the doctor has been scheduled for Monday at ten thirty in the morning.',
 'Please hold for a moment while I transfer your call to the concerned department.',
 'The train to Bengaluru will depart from platform number six at a quarter past nine.',
 'If you have any further questions, feel free to reach out to our customer support team at any time.',
 'She opened the old wooden door slowly and stepped into the quiet, dusty library.',
 'Your loan application has been approved, and the amount will be credited within two working days.',
 'To reset your password, click the link we have sent to your registered email address.',
 'The quarterly results exceeded expectations, driven by strong growth in the southern region.',
 'Kindly note that the office will remain closed on Friday on account of the festival.',
 'He paused for a second, took a deep breath, and then answered the question calmly.',
 'We are sorry for the inconvenience caused and assure you that the issue will be resolved today.',
 'Turn left at the next signal, continue for two hundred metres, and the hospital will be on your right.',
 'This call may be recorded for quality and training purposes.',
 'The total amount payable, including taxes, comes to four thousand eight hundred and fifty rupees.',
 'Can you hear me clearly, or should I repeat the last part once again?',
 'Our records show that the previous instalment was received on the twelfth of last month.',
 'Thank you for your patience. Have a wonderful day ahead.']
ES = ['Buenos días, le llamamos del banco para confirmar el pago de dos mil quinientos pesos realizado ayer.',
 'Su pedido será entregado mañana por la tarde, por favor mantenga su teléfono disponible.',
 'Gracias por su paciencia, nuestro equipo ha registrado su solicitud y le contactará en breve.',
 'Lamentamos las molestias ocasionadas y le aseguramos que el problema quedará resuelto hoy mismo.']
DE = ['Guten Morgen, wir rufen wegen Ihrer Kreditkartenabrechnung für den Monat September an.',
 'Ihr Termin beim Arzt wurde für Montag um halb elf Uhr vormittags eingetragen.',
 'Bitte bleiben Sie einen Moment in der Leitung, ich verbinde Sie mit der zuständigen Abteilung.',
 'Vielen Dank für Ihre Geduld, wir wünschen Ihnen noch einen schönen Tag.']
LANG_NAME = {"en": "English", "es": "Spanish", "de": "German"}
ALL = [("en", f"en{i:02d}", t) for i, t in enumerate(EN)] + [("es", f"es{i:02d}", t) for i, t in enumerate(ES)] + [("de", f"de{i:02d}", t) for i, t in enumerate(DE)]
import os
REF_WAV = (os.environ.get("QWEN3TTS_ROOT") or os.path.dirname(os.path.abspath(__file__))) + "/samples/voice_ref.wav"
REF_TEXT = EN[0]
