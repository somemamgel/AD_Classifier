server_url = 'http://127.0.0.1:9000/api/prediction';  // 默认请求地址

var res_text_dict = {
    akiec: '光化性角化病/上皮内癌/鲍恩氏病（akiec）。',
    bcc: '基底细胞癌（bcc）。发生转移率低，比较偏向于良性。',
    bkl: '良性角化病（bkl）。',
    df: '皮肤纤维瘤（df）。是成纤维细胞或组织细胞灶性增生引致的一种真皮内的良性肿瘤。',
    mel: '黑色素瘤（mel）。恶性黑色素瘤，是黑色素细胞来源的一种高度恶性的肿瘤。',
    nv: '黑色素痣（nv）。是由一群良性的黑色素细胞，聚集在表皮与真皮的交界产生的。',
    vasc: '血管病变（vasc）。'
};


// 获取上传图片的路径
function getFileUri(obj) {
    let uri;
    uri = window.URL.createObjectURL(obj.files.item(0));
    return uri;
}

// Vue 处理请求
const app = Vue.createApp({
    data() {
        return {
            filename: '',
            image_path: '',
            ad_value: null,
            mci_value: null,
            nc_value: null,
        }
    },
    methods: {
        handleFileUpload(event) {
            let ftype = this.$refs.file.files[0].type;
            let fpath = getFileUri(this.$refs.file);
            if (true/*ftype == 'image/jpeg' || ftype == 'image/png'*/) {
                this.filename = this.$refs.file.files[0].name;
                this.file = this.$refs.file.files[0];
                this.image_path = fpath;
                // readFile(this.$refs.file.files[0]);
            }
            else {
                alert('请选择 .nii .nii.gz 文件')
                this.filename = this.image_path = '';
            }
        },
        handleSubmitRequests(event) {
            if (this.file) {
                let file = this.file;
                let param = new FormData();
                param.append('image', file, file.name);
                axios({
                    method: 'post',
                    url: server_url,
                    data: param,
                    headers: {}
                }).then((response) => {
                    let result = response.data;
                    if (result[0] === 'prob' && Object.keys(result[1]).length == 3) {
                        console.log("data received from api!");
                        let r = result[1];
                        this.ad_value = (r['ad'] * 100).toFixed(4) + '%';
                        this.mci_value = (r['mci'] * 100).toFixed(4) + '%';
                        this.nc_value = (r['nc'] * 100).toFixed(4) + '%';
                        let max = 0;
                        let index;
                        for (let i in r) {
                            if (r[i] > max) {
                                index = i;
                                max = r[i];
                            }
                        }
                        this.result_text = (r[index] * 100).toFixed(2) + '%' + ' - ' + res_text_dict[index];
                    }
                    else {
                        console.log("faild!");
                        this.ad_value = this.nc_value = null;
                    }
                });
            }
            else {
                alert("请选择文件");
            }
        },
        handleClear() {
            this.file = null;
            this.filename = this.image_path = '';
            this.ad_value = this.nc_value = null;
            this.result_text = '';
        }
    }
}).mount("#predict-body");



// app.component('my-snackbar', {
//     data() {
//         return {
//             message: 'test'
//         }
//     },
//     template: `
//         <span class="tag is-danger is-medium">
//             {{ message }}
//             <button class="delete is-small"></button>
//         </span>
//         `
// })