from OpenSSL import crypto
import os

def create_self_signed_cert():
    # สร้าง key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)

    # สร้าง certificate
    cert = crypto.X509()
    cert.get_subject().C = "TH"
    cert.get_subject().ST = "Bangkok"
    cert.get_subject().L = "Bangkok"
    cert.get_subject().O = "Sign Language Detection"
    cert.get_subject().OU = "Development"
    cert.get_subject().CN = "localhost"

    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # หมดอายุใน 1 ปี
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # เขียนไฟล์ certificate และ private key
    with open("certs/cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("certs/key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

if __name__ == '__main__':
    if not os.path.exists('certs'):
        os.makedirs('certs')
    create_self_signed_cert()
    print("สร้าง SSL certificate เรียบร้อยแล้ว") 