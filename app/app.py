from flask import Flask, render_template
import tensorflow as tf
import json
import numpy as np
import model, sample, encoder
import os
from pathlib import Path
from urllib.request import urlretrieve
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired

app = Flask(__name__)
path = Path(__file__).parent.parent

app.config['SECRET_KEY'] = 'I-dont-understand-why-i-need-this'

def download_file(url, dest):
    if dest.exists(): return
    else:
        print(f'Downloading {dest}')
        urlretrieve(url, dest)

def download_req_files():
    root_url = "https://storage.googleapis.com/gpt-2/models/"
    files = ["checkpoint", "encoder.json", "hparams.json", "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta", "vocab.bpe"]
    for f in files:
        source = root_url + '117M/' + f
        dest = path/'models'/'117M'/f
        dest.parent.mkdir(parents=True, exist_ok=True)
        download_file(source, dest)
    return True


@app.route("/", methods=('GET', 'POST'))
def main():
    form = InputForm()
    output_text = None
    input_text = None
    if form.validate_on_submit():
        input_text = form.input_text.data
        output_text = get_interact(input_text=input_text)
    return render_template('index.html', form=form, input_text=input_text,
                           output_text=output_text)

def get_interact(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=None,
    length=None,
    temperature=1,
    top_k=0,
    input_text='Say whatever you want'
):

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(path/'models'/model_name/'hparams.json') as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(path/'models'/model_name)
        saver.restore(sess, ckpt)

        while True:
            context_tokens = enc.encode(input_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i]).split('<|endoftext|>')[0]
                    return text


class InputForm(FlaskForm):
    input_text = TextAreaField('Input text')
    submit = SubmitField()

if __name__ == "__main__":
    download_req_files()
    app.run()
